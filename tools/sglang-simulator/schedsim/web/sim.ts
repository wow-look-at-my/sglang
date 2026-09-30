// Visual scheduler simulator: the old prefill-priority rule and the fork's
// time-sharing balancer, side by side, fed the same traffic.
//
// The cost model and the two scheduling rules are ports of internal/sim and
// python/sglang/srt/managers/scheduler_components/prefill_decode_balancer.py,
// priced with the calibration schedsim fitted to the Qwen3.8-Flash-Next log
// (see docs/replaying-a-serving-log.md). One file, no imports, so the compiled
// sim.js runs from a file:// open.

// ---------------------------------------------------------------- cost model

interface Calibration {
  chunk: number;
  prefillBase: number; // s per prefill batch
  perToken: number; // s per token
  perTokenCtx: number; // s per token per context token
  perTokenCtxSq: number;
  decodeBase: number; // s per decode step
  decodePerReq: number;
  decodePerTokenCtx: number;
  acceptMean: number; // tokens per decode step per request (speculative)
  poolTokens: number; // device KV pool
  reloadPerToken: number; // host tier reload, s per token
}

// schedsim's fit on the 28-hour log: "prefill 10.0 ms + 65.874 us/token +
// 7.43e-11/token/ctx + 5.54e-16/token/ctx^2; decode 13.269 ms + 1.5 ms/req +
// 6e-09 s/token-ctx (accept 2.78); pool 1398667 tokens".
const LOG_CALIBRATION: Calibration = {
  chunk: 4096,
  prefillBase: 10e-3,
  perToken: 65.874e-6,
  perTokenCtx: 7.43e-11,
  perTokenCtxSq: 5.54e-16,
  decodeBase: 13.269e-3,
  decodePerReq: 1.5e-3,
  decodePerTokenCtx: 6e-9,
  acceptMean: 2.78,
  poolTokens: 1398667,
  reloadPerToken: 0.5e-6,
};

function prefillSeconds(c: Calibration, items: { tokens: number; midCtx: number }[]): number {
  if (items.length === 0) return 0;
  let t = c.prefillBase;
  for (const it of items) {
    const m = it.midCtx;
    t += it.tokens * (c.perToken + c.perTokenCtx * m + c.perTokenCtxSq * m * m);
  }
  return t;
}

function decodeSeconds(c: Calibration, batch: number, sumCtx: number): number {
  if (batch <= 0) return 0;
  return c.decodeBase + c.decodePerReq * (batch - 1) + c.decodePerTokenCtx * sumCtx;
}

// ---------------------------------------------------------------- balancer

// PrefillDecodeBalancer: measured GPU seconds of prefill against decode while
// both have work; prefill is deferred while it is ahead. Time is charged at
// completion, from the previous completion, because the overlap scheduler
// picks batch N+1 while N still runs. A burst of at most one chunk's tokens
// runs before decode must catch up; a chunk continuing a long prompt is
// bounded in GPU seconds at its marginal (context-inflated) rate.
class Balancer {
  debt = 0;
  unsettledPf = 0;
  unsettledDc = 0;
  unsettledN = 0;
  lastDecode = 0;
  burstUsed = 0;
  extendSecond = 0;
  extendTokens = 0;
  lastPrefillSecond = 0;
  lastPrefillTokens = 0;
  private inFlight: { isPrefill: boolean; tokens: number; rows: number }[] = [];
  private busy = 0;
  constructor(public burstTokens: number, public piggybackCredit: boolean) {}

  prefillSecondsPerToken(): number {
    return this.extendTokens === 0 ? 0 : this.extendSecond / this.extendTokens;
  }
  private marginalSecondsPerToken(): number {
    return this.lastPrefillTokens === 0 ? 0 : this.lastPrefillSecond / this.lastPrefillTokens;
  }
  private prefillInFlight(): boolean {
    return this.inFlight.some((f) => f.isPrefill);
  }

  shouldDeferPrefill(prefillPending: boolean, decodeRunnable: boolean, continuesChunk: boolean): boolean {
    if (!prefillPending || !decodeRunnable) {
      this.debt = 0;
      this.unsettledPf = this.unsettledDc = 0;
      this.unsettledN = 0;
      this.burstUsed = 0;
      return false;
    }
    if (this.unsettledN > 0) {
      // At most the last decode batch overshoots into the next decision.
      this.debt = Math.max(this.debt + this.unsettledPf - this.unsettledDc, -this.lastDecode);
      this.unsettledPf = this.unsettledDc = 0;
      this.unsettledN = 0;
    }
    if (this.debt <= 0 && !this.prefillInFlight()) this.burstUsed = 0;
    if (this.burstTokens === 0) return this.debt > 0;
    if (continuesChunk) return this.burstUsed > 0;
    return this.burstUsed >= this.burstTokens;
  }

  // -1 means no bound.
  prefillTokenBudget(continuesChunk: boolean): number {
    if (this.burstTokens === 0) return -1;
    const average = this.prefillSecondsPerToken();
    const marginal = this.marginalSecondsPerToken();
    if (continuesChunk && average > 0 && marginal > 0) {
      return Math.max(0, Math.floor((this.burstTokens * average) / marginal));
    }
    if (this.burstUsed === 0) return -1;
    return this.burstTokens - this.burstUsed;
  }

  onLaunch(isPrefill: boolean, tokens: number, rows: number, now: number): void {
    if (this.inFlight.length === 0) this.busy = now;
    this.inFlight.push({ isPrefill, tokens, rows });
    if (isPrefill) this.burstUsed += tokens;
  }

  onFinish(now: number): void {
    if (this.inFlight.length === 0) return;
    const el = now - this.busy;
    this.busy = now;
    const f = this.inFlight.shift()!;
    let piggy = 0;
    if (f.rows > 0 && this.piggybackCredit) piggy = Math.min(f.rows * this.prefillSecondsPerToken(), el);
    if (f.isPrefill && f.tokens > 0) {
      this.extendSecond += el;
      this.extendTokens += f.tokens;
      this.lastPrefillSecond = el;
      this.lastPrefillTokens = f.tokens;
    }
    if (f.isPrefill) this.unsettledPf += el - piggy;
    else {
      this.unsettledDc += el;
      this.lastDecode = el;
    }
    this.unsettledN++;
  }
}

// ---------------------------------------------------------------- KV pool

// The device pool holds running requests' reservations and finished
// conversations' prefixes. Admission evicts the least recently used idle
// prefixes, whole, when the free space is short; an optional host tier keeps
// what was evicted, and a hit there is reloaded instead of recomputed.
class Pool {
  entries = new Map<number, { dev: number; host: number; lastUse: number }>();
  inflight = 0;
  devCached = 0;
  hostUsed = 0;
  recomputes = 0;
  recomputeTokens = 0;
  reloads = 0;
  evictions = 0;
  private seen = new Set<number>();
  constructor(public devCap: number, public hostCap: number) {}

  free(): number {
    return this.devCap - this.inflight - this.devCached;
  }
  usage(): number {
    return (this.inflight + this.devCached) / this.devCap;
  }
  lookup(conv: number, input: number): { dev: number; host: number } {
    const e = this.entries.get(conv);
    if (!e) return { dev: 0, host: 0 };
    if (e.dev > 0) return { dev: Math.min(e.dev, input - 1), host: 0 };
    return { dev: 0, host: Math.min(e.host, input - 1) };
  }
  // Frees short tokens by evicting idle prefixes other than conv's own.
  evictFor(conv: number, short: number): boolean {
    const cands = [...this.entries.entries()]
      .filter(([c, e]) => c !== conv && e.dev > 0)
      .sort((a, b) => a[1].lastUse - b[1].lastUse);
    let got = 0;
    for (const [c, e] of cands) {
      if (got >= short) break;
      got += e.dev;
      this.devCached -= e.dev;
      this.evictions++;
      if (this.hostCap - this.hostUsed >= e.dev) {
        e.host = e.dev;
        this.hostUsed += e.dev;
      } else if (e.host === 0) {
        this.entries.delete(c);
      }
      e.dev = 0;
    }
    return got >= short;
  }
  // Takes conv's reservation. The resident prefix stays in device memory
  // for the run (it moves from the cached total to the reservation), a host
  // hit is reloaded into device memory, and the rest is computed: device
  // space is needed for everything but the resident prefix.
  admit(conv: number, input: number, out: number, devHit: number, hostHit: number, now: number): void {
    if (this.seen.has(conv) && 2 * (devHit + hostHit) < input) {
      this.recomputes++;
      this.recomputeTokens += input - devHit - hostHit;
    }
    this.seen.add(conv);
    const e = this.entries.get(conv);
    if (e) {
      this.devCached -= e.dev;
      if (e.host > 0) {
        this.hostUsed -= e.host;
        this.reloads++;
      }
      this.entries.delete(conv);
    }
    this.inflight += input + out;
  }
  // Returns the reservation and keeps the whole context cached: it is the
  // memory the request already held, now evictable.
  release(conv: number, ctx: number, reserved: number, now: number): void {
    this.inflight -= reserved;
    const dev = Math.min(ctx, reserved);
    this.devCached += dev;
    this.entries.set(conv, { dev, host: 0, lastUse: now });
  }
  touch(conv: number, now: number): void {
    const e = this.entries.get(conv);
    if (e) e.lastUse = now;
  }
}

// ---------------------------------------------------------------- requests

type Kind = "agent" | "cold";

interface Req {
  id: number;
  conv: number;
  kind: Kind;
  arrival: number;
  input: number;
  out: number;
  prefix: number; // prompt tokens done (cached or computed)
  outDone: number;
  reserved: number;
  reloadToks: number;
  hitAtAdmit: number;
  state: "waiting" | "running" | "done";
  prefillStart: number;
  firstTok: number;
  finish: number;
  lastDelivery: number;
  tag: string;
}

interface Batch {
  start: number;
  end: number;
  kind: "prefill" | "decode" | "mixed";
  tokens: number; // prefill tokens
  rows: number; // decode rows
  cold: boolean; // carries a cold prompt's chunk
}

// ---------------------------------------------------------------- engine

interface EngineConfig {
  mode: "old" | "new";
  cal: Calibration;
  maxRunning: number;
  hostMul: number;
  mixedChunk: boolean;
  // cede: while a long prompt is being chunked, waiting requests short enough
  // to finish inside half a chunk ride along (the fork's cede budget); the
  // upstream rule gives the chunked prompt the whole chunk every time.
  cede: boolean;
}

interface Delivered {
  t: number;
  n: number;
  conv: number;
}

class Engine {
  now = 0;
  waiting: Req[] = [];
  running: Req[] = [];
  chunked: Req | null = null;
  requests: Req[] = [];
  batches: Batch[] = [];
  deliveries: Delivered[] = [];
  pool: Pool;
  bal: Balancer | null;
  longestStall = 0;
  stallAt = 0;
  private lastDeliveryAny = -1;
  busyPrefill = 0;
  busyDecode = 0;
  private nextId = 1;
  private feed: Feed;

  constructor(public cfg: EngineConfig, feed: Feed) {
    this.feed = feed;
    this.pool = new Pool(cfg.cal.poolTokens, Math.floor(cfg.cal.poolTokens * cfg.hostMul));
    this.bal = cfg.mode === "new" ? new Balancer(cfg.cal.chunk, true) : null;
  }

  // Warms a conversation's prefix as if it had run before the clock started.
  warm(conv: number, tokens: number): void {
    this.pool.release(conv, tokens, tokens, 0);
  }

  add(kind: Kind, conv: number, arrival: number, input: number, out: number, tag = ""): Req {
    const r: Req = {
      id: this.nextId++, conv, kind, arrival, input, out, prefix: 0, outDone: 0, reserved: 0, reloadToks: 0, hitAtAdmit: 0,
      state: "waiting", prefillStart: -1, firstTok: -1, finish: -1, lastDelivery: -1, tag,
    };
    this.requests.push(r);
    this.waiting.push(r);
    this.waiting.sort((a, b) => a.arrival - b.arrival || a.id - b.id);
    return r;
  }

  // Runs batches until the clock reaches t.
  advanceTo(t: number): void {
    let guard = 0;
    while (this.now < t && guard++ < 200000) {
      const b = this.nextBatch();
      if (!b) {
        // Idle: jump to the next arrival.
        const next = this.waiting.find((r) => r.arrival > this.now);
        this.now = next ? Math.min(next.arrival, t) : t;
        continue;
      }
      this.run(b);
    }
  }

  private arrived(): Req[] {
    return this.waiting.filter((r) => r.arrival <= this.now);
  }

  private nextBatch(): { kind: "prefill" | "decode"; items: { req: Req; tokens: number }[]; rows: Req[] } | null {
    const arrived = this.arrived();
    const prefillPending = arrived.length > 0 || this.chunked !== null;
    const decodeRunnable = this.running.some((r) => r.state === "running" && r.prefix >= r.input);
    if (!prefillPending && !decodeRunnable) return null;
    let doPrefill = prefillPending;
    let budget = -1;
    if (this.bal) {
      const continues = this.chunked !== null;
      if (prefillPending && decodeRunnable && this.bal.shouldDeferPrefill(true, true, continues)) doPrefill = false;
      else if (prefillPending) budget = this.bal.prefillTokenBudget(continues);
      if (!prefillPending) this.bal.shouldDeferPrefill(false, decodeRunnable, false);
    }
    if (doPrefill) {
      const items = this.formPrefill(budget);
      if (items.length > 0) {
        const rows = this.cfg.mixedChunk
          ? this.running.filter((r) => r.state === "running" && r.prefix >= r.input && !items.some((it) => it.req === r))
          : [];
        return { kind: "prefill", items, rows };
      }
    }
    if (decodeRunnable) {
      return { kind: "decode", items: [], rows: this.running.filter((r) => r.state === "running" && r.prefix >= r.input) };
    }
    return null;
  }

  private formPrefill(budget: number): { req: Req; tokens: number }[] {
    const chunk = this.cfg.cal.chunk;
    let cap = budget < 0 ? chunk : Math.min(chunk, budget);
    if (cap <= 0) return [];
    const items: { req: Req; tokens: number }[] = [];
    if (this.chunked) {
      const r = this.chunked;
      if (this.cfg.cede) {
        // Up to half the chunk goes to waiting requests that fit in it whole.
        let reserve = Math.floor(cap / 2);
        for (const w of this.arrived()) {
          if (reserve <= 0 || this.running.length >= this.cfg.maxRunning) break;
          const hit = this.pool.lookup(w.conv, w.input);
          const own = w.input - hit.dev - hit.host;
          if (own > reserve) continue;
          if (!this.admit(w)) continue;
          items.push({ req: w, tokens: own });
          reserve -= own;
          cap -= own;
        }
      }
      const left = r.input - r.prefix;
      items.push({ req: r, tokens: Math.min(left, Math.max(cap, 1)) });
      return items;
    }
    for (const r of this.arrived()) {
      if (cap <= 0) break;
      if (this.running.length >= this.cfg.maxRunning) break;
      if (!this.admit(r)) break;
      const left = r.input - r.prefix;
      const tokens = Math.min(left, cap);
      items.push({ req: r, tokens });
      cap -= tokens;
      if (tokens < left) {
        this.chunked = r;
        break;
      }
    }
    return items;
  }

  private admit(r: Req): boolean {
    const hit = this.pool.lookup(r.conv, r.input);
    const need = r.input - hit.dev + r.out;
    if (this.pool.free() < need) {
      if (!this.pool.evictFor(r.conv, need - this.pool.free())) return false;
    }
    this.pool.admit(r.conv, r.input, r.out, hit.dev, hit.host, this.now);
    r.prefix = hit.dev + hit.host;
    r.hitAtAdmit = r.prefix;
    r.reloadToks = hit.host;
    r.reserved = r.input + r.out;
    r.state = "running";
    r.prefillStart = this.now;
    this.waiting = this.waiting.filter((x) => x !== r);
    this.running.push(r);
    return true;
  }

  private run(b: { kind: "prefill" | "decode"; items: { req: Req; tokens: number }[]; rows: Req[] }): void {
    const cal = this.cfg.cal;
    const start = this.now;
    let seconds: number;
    let tokens = 0;
    let cold = false;
    if (b.kind === "prefill") {
      const priced = b.items.map((it) => ({ tokens: it.tokens, midCtx: it.req.prefix + it.tokens / 2 }));
      // Mixed chunk: each running request rides along as one extend token
      // at the chunk's rate and yields one plain token.
      for (const r of b.rows) priced.push({ tokens: 1, midCtx: r.input + r.outDone });
      seconds = prefillSeconds(cal, priced);
      for (const it of b.items) {
        tokens += it.tokens;
        seconds += it.req.reloadToks * cal.reloadPerToken;
        it.req.reloadToks = 0;
        if (it.req.kind === "cold") cold = true;
      }
      this.busyPrefill += seconds;
    } else {
      seconds = decodeSeconds(cal, b.rows.length, b.rows.reduce((s, r) => s + r.input + r.outDone, 0));
      this.busyDecode += seconds;
    }
    this.bal?.onLaunch(b.kind === "prefill", tokens, b.rows.length, start);
    this.now = start + seconds;
    const end = this.now;
    this.batches.push({ start, end, kind: b.kind === "prefill" ? (b.rows.length ? "mixed" : "prefill") : "decode", tokens, rows: b.rows.length, cold });
    if (this.batches.length > 20000) this.batches.splice(0, 5000);
    for (const it of b.items) {
      const r = it.req;
      r.prefix += it.tokens;
      if (r.prefix >= r.input) {
        if (this.chunked === r) this.chunked = null;
        this.deliver(r, end, 1);
      }
    }
    const perRow = b.kind === "decode" ? cal.acceptMean : 1;
    for (const r of b.rows) this.deliver(r, end, perRow);
    this.bal?.onFinish(end);
  }

  private deliver(r: Req, at: number, n: number): void {
    const give = Math.min(n, r.out - r.outDone);
    if (give <= 0) return;
    if (r.firstTok < 0) r.firstTok = at;
    r.outDone += give;
    this.deliveries.push({ t: at, n: give, conv: r.conv });
    if (this.deliveries.length > 200000) this.deliveries.splice(0, 50000);
    if (r.kind === "agent") {
      const gap = at - (r.lastDelivery >= 0 ? r.lastDelivery : r.arrival);
      if (gap > this.longestStall) {
        this.longestStall = gap;
        this.stallAt = at;
      }
    }
    r.lastDelivery = at;
    this.pool.touch(r.conv, at);
    if (r.outDone >= r.out) {
      r.state = "done";
      r.finish = at;
      this.running = this.running.filter((x) => x !== r);
      this.pool.release(r.conv, r.input + r.outDone, r.reserved, at);
      r.reserved = 0;
      this.feed.onFinish(this, r, at);
    }
  }

  // The longest wait an agent stream is in right now: since its last token,
  // or since its turn arrived if it has had none.
  currentStall(): number {
    let worst = 0;
    for (const r of this.requests) {
      if (r.kind !== "agent" || r.state === "done" || r.arrival > this.now) continue;
      const since = r.lastDelivery >= 0 ? r.lastDelivery : r.arrival;
      worst = Math.max(worst, this.now - since);
    }
    return worst;
  }

  // Tokens delivered to agent streams over the last w seconds.
  rateOver(w: number, conv?: number): number {
    const from = this.now - w;
    let n = 0;
    for (let i = this.deliveries.length - 1; i >= 0; i--) {
      const d = this.deliveries[i];
      if (d.t < from) break;
      if (conv === undefined || d.conv === conv) n += d.n;
    }
    return n / w;
  }
}

// ---------------------------------------------------------------- traffic

// Mulberry32: one seed gives both engines the same draws.
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface Stream {
  conv: number;
  ctx: number;
  draw: () => number;
  label: string;
}

interface TrafficParams {
  seed: number;
  agents: number;
  ctxMin: number;
  ctxMax: number;
  newMin: number;
  newMax: number;
  outMin: number;
  outMax: number;
  thinkMin: number;
  thinkMax: number;
  coldEvery: number; // mean seconds between random cold prompts, 0 = none
  coldMin: number;
  coldMax: number;
  shared: number;
}

// Feed is one side's copy of the traffic. Each stream draws from its own
// generator, so both engines see the same turn sizes and think times whatever
// order they finish in; the random cold prompts are a precomputed schedule
// both engines share.
class Feed {
  streams = new Map<number, Stream>();
  coldSchedule: { at: number; input: number; out: number }[] = [];
  private nextConv = 1;
  constructor(public p: TrafficParams) {
    const r = rng(p.seed);
    for (let i = 0; i < p.agents; i++) {
      const conv = this.nextConv++;
      const ctx = Math.floor(p.ctxMin + r() * (p.ctxMax - p.ctxMin));
      this.streams.set(conv, { conv, ctx, draw: rng(p.seed * 7919 + conv), label: `agent ${i + 1}` });
    }
    if (p.coldEvery > 0) {
      const cr = rng(p.seed * 31 + 5);
      let t = 0;
      for (let k = 0; k < 400; k++) {
        t += -Math.log(1 - cr()) * p.coldEvery;
        this.coldSchedule.push({ at: t, input: Math.floor(p.coldMin + cr() * (p.coldMax - p.coldMin)), out: 300 });
      }
    }
  }
  seed(e: Engine): void {
    for (const s of this.streams.values()) {
      // The system prompt is part of every cached prefix.
      e.warm(s.conv, this.p.shared + s.ctx);
      this.turn(e, s, this.uni(s.draw, 0.2, 2));
    }
    for (const c of this.coldSchedule) e.add("cold", this.nextConv + 1000 + this.coldSchedule.indexOf(c), c.at, c.input, c.out, "random cold");
  }
  private uni(d: () => number, a: number, b: number): number {
    return a + d() * (b - a);
  }
  turn(e: Engine, s: Stream, at: number): void {
    const p = this.p;
    const n = Math.floor(this.uni(s.draw, p.newMin, p.newMax));
    const out = Math.floor(this.uni(s.draw, p.outMin, p.outMax));
    e.add("agent", s.conv, at, p.shared + s.ctx + n, out);
  }
  onFinish(e: Engine, r: Req, now: number): void {
    const s = this.streams.get(r.conv);
    if (!s) return;
    s.ctx = r.input - this.p.shared + r.outDone;
    this.turn(e, s, now + this.uni(s.draw, this.p.thinkMin, this.p.thinkMax));
  }
  injectCold(e: Engine, at: number, input: number, out: number, tag: string): void {
    e.add("cold", 5000 + Math.floor(at * 1000), at, input, out, tag);
  }
  injectFollowUp(e: Engine, at: number, conv: number): void {
    const s = this.streams.get(conv);
    if (s) this.turn(e, s, at);
  }
  // A new conversation that then continues as an agent.
  injectNewAgent(e: Engine, at: number, input: number): void {
    const conv = this.nextConv++;
    const s: Stream = { conv, ctx: input, draw: rng(this.p.seed * 7919 + conv), label: `agent ${this.streams.size + 1}` };
    this.streams.set(conv, s);
    e.add("cold", conv, at, this.p.shared + input, 300, "new conversation");
  }
}

// ---------------------------------------------------------------- side

class Side {
  engine: Engine;
  feed: Feed;
  constructor(public mode: "old" | "new", traffic: TrafficParams, cal: Calibration, maxRunning: number, hostMul: number, mixed: boolean, cede: boolean) {
    this.feed = new Feed(traffic);
    this.engine = new Engine({ mode, cal, maxRunning, hostMul, mixedChunk: mixed, cede }, this.feed);
    this.feed.seed(this.engine);
  }
}

// ---------------------------------------------------------------- headless check

// `node sim.js` runs a short comparison and prints it; the browser skips
// this, and so does a test that loads the file as a module.
declare const process: any;
declare const require: any;
declare const module: any;
if (typeof document === "undefined" && typeof require !== "undefined" && require.main === module) {
  const traffic: TrafficParams = {
    seed: 7, agents: 5, ctxMin: 100000, ctxMax: 250000, newMin: 200, newMax: 1400, outMin: 200, outMax: 800,
    thinkMin: 1, thinkMax: 5, coldEvery: 90, coldMin: 150000, coldMax: 400000, shared: 12288,
  };
  for (const mode of ["old", "new"] as const) {
    const side = new Side(mode, traffic, LOG_CALIBRATION, 16, mode === "new" ? 4 : 0, mode === "new", mode === "new");
    side.engine.advanceTo(600);
    const e = side.engine;
    const cold = e.requests.filter((r) => r.kind === "cold" && r.firstTok >= 0);
    const ttft = cold.length ? cold.reduce((s, r) => s + r.firstTok - r.arrival, 0) / cold.length : 0;
    console.log(
      `${mode}: longest stall ${e.longestStall.toFixed(1)} s, agent tok/s ${(e.rateOver(600) ).toFixed(1)}, ` +
        `cold TTFT ${ttft.toFixed(1)} s over ${cold.length} prompts, recomputes ${e.pool.recomputes}, batches ${e.batches.length}`,
    );
  }
}

// ---------------------------------------------------------------- UI

interface SideView {
  side: Side;
  timeline: HTMLCanvasElement;
  rates: HTMLCanvasElement;
  stats: HTMLElement;
  queue: HTMLElement;
  streams: HTMLElement;
  history: number[][]; // per-stream tok/s samples
  aggHistory: number[];
}

function el<T extends HTMLElement>(id: string): T {
  const e = document.getElementById(id);
  if (!e) throw new Error("missing #" + id);
  return e as T;
}

function num(id: string): number {
  return parseFloat((el<HTMLInputElement>(id)).value);
}

if (typeof document !== "undefined") {
  const views: SideView[] = [];
  let simTime = 0;
  let running = false;
  let speed = 10;
  let lastFrame = 0;
  const windowSec = 60;
  const sampleEvery = 0.5;
  let lastSample = 0;

  function traffic(): TrafficParams {
    return {
      seed: num("seed"), agents: num("agents"), ctxMin: num("ctxMin") * 1000, ctxMax: num("ctxMax") * 1000,
      newMin: 200, newMax: 1400, outMin: 200, outMax: 800, thinkMin: num("thinkMin"), thinkMax: num("thinkMax"),
      coldEvery: num("coldEvery"), coldMin: num("coldMin") * 1000, coldMax: num("coldMax") * 1000, shared: 12288,
    };
  }

  function reset(): void {
    const t = traffic();
    const cal = { ...LOG_CALIBRATION, chunk: num("chunk"), poolTokens: num("pool") * 1000 };
    const maxRunning = num("maxRunning");
    const hostNew = (el<HTMLInputElement>("hicache")).checked ? 4 : 0;
    const mixed = (el<HTMLInputElement>("mixed")).checked;
    const cede = (el<HTMLInputElement>("cede")).checked;
    views.length = 0;
    for (const mode of ["old", "new"] as const) {
      const side = new Side(mode, t, cal, maxRunning, mode === "new" ? hostNew : 0, mode === "new" && mixed, mode === "new" && cede);
      views.push({
        side,
        timeline: el<HTMLCanvasElement>(`${mode}-timeline`),
        rates: el<HTMLCanvasElement>(`${mode}-rates`),
        stats: el(`${mode}-stats`),
        queue: el(`${mode}-queue`),
        streams: el(`${mode}-streams`),
        history: [],
        aggHistory: [],
      });
    }
    simTime = 0;
    lastSample = 0;
    el("clock").textContent = "0.0 s";
    render();
  }

  function step(dt: number): void {
    simTime += dt;
    for (const v of views) v.side.engine.advanceTo(simTime);
    while (lastSample + sampleEvery <= simTime) {
      lastSample += sampleEvery;
      for (const v of views) {
        const e = v.side.engine;
        const per: number[] = [];
        for (const s of v.side.feed.streams.values()) per.push(e.rateOver(2, s.conv));
        v.history.push(per);
        v.aggHistory.push(e.rateOver(2));
        const keep = Math.ceil(windowSec / sampleEvery);
        if (v.history.length > keep) {
          v.history.shift();
          v.aggHistory.shift();
        }
      }
    }
    el("clock").textContent = simTime.toFixed(1) + " s";
  }

  function frame(ts: number): void {
    if (running) {
      const wall = lastFrame ? Math.min((ts - lastFrame) / 1000, 0.1) : 0;
      lastFrame = ts;
      step(wall * speed);
      render();
    }
    requestAnimationFrame(frame);
  }

  function color(kind: Batch["kind"], cold: boolean): string {
    if (kind === "decode") return "#3b82f6";
    if (kind === "mixed") return "#8b5cf6";
    return cold ? "#ea580c" : "#f59e0b";
  }

  function drawTimeline(v: SideView): void {
    const c = v.timeline;
    const ctx = c.getContext("2d")!;
    const W = c.width, H = c.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#0f172a";
    ctx.fillRect(0, 0, W, H);
    const from = simTime - windowSec;
    const x = (t: number) => ((t - from) / windowSec) * W;
    const e = v.side.engine;
    for (let i = e.batches.length - 1; i >= 0; i--) {
      const b = e.batches[i];
      if (b.end < from) break;
      const x0 = Math.max(0, x(b.start)), x1 = Math.min(W, x(b.end));
      ctx.fillStyle = color(b.kind, b.cold);
      ctx.fillRect(x0, 6, Math.max(x1 - x0, 0.5), H - 12);
    }
    ctx.fillStyle = "#94a3b8";
    ctx.font = "11px system-ui";
    for (let s = 10; s <= windowSec; s += 10) {
      const xx = x(simTime - windowSec + s);
      ctx.fillRect(xx, H - 4, 1, 4);
      ctx.fillText(`-${windowSec - s}s`, xx + 2, H - 6);
    }
  }

  function drawRates(v: SideView): void {
    const c = v.rates;
    const ctx = c.getContext("2d")!;
    const W = c.width, H = c.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#0f172a";
    ctx.fillRect(0, 0, W, H);
    const n = Math.ceil(windowSec / sampleEvery);
    const maxRate = Math.max(300, ...views.flatMap((vv) => vv.aggHistory));
    const y = (r: number) => H - 4 - (r / maxRate) * (H - 8);
    ctx.strokeStyle = "#334155";
    ctx.beginPath();
    ctx.moveTo(0, y(100));
    ctx.lineTo(W, y(100));
    ctx.stroke();
    ctx.fillStyle = "#64748b";
    ctx.font = "10px system-ui";
    ctx.fillText("100 tok/s", 4, y(100) - 2);
    const streams = [...v.side.feed.streams.values()];
    const palette = ["#22c55e", "#06b6d4", "#eab308", "#f472b6", "#a3e635", "#fb923c", "#c084fc", "#2dd4bf"];
    streams.forEach((s, si) => {
      ctx.strokeStyle = palette[si % palette.length];
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let k = 0; k < v.history.length; k++) {
        const xx = ((k + (n - v.history.length)) / n) * W;
        const r = v.history[k][si] ?? 0;
        if (k === 0) ctx.moveTo(xx, y(r));
        else ctx.lineTo(xx, y(r));
      }
      ctx.stroke();
    });
    ctx.strokeStyle = "#f8fafc";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let k = 0; k < v.aggHistory.length; k++) {
      const xx = ((k + (n - v.aggHistory.length)) / n) * W;
      if (k === 0) ctx.moveTo(xx, y(v.aggHistory[k]));
      else ctx.lineTo(xx, y(v.aggHistory[k]));
    }
    ctx.stroke();
  }

  function render(): void {
    for (const v of views) {
      const e = v.side.engine;
      drawTimeline(v);
      drawRates(v);
      const cold = e.requests.filter((r) => r.kind === "cold");
      const served = cold.filter((r) => r.firstTok >= 0);
      const ttft = served.length ? served.reduce((s, r) => s + r.firstTok - r.arrival, 0) / served.length : 0;
      const busy = e.busyPrefill + e.busyDecode;
      const cur = e.currentStall();
      const worst = Math.max(e.longestStall, cur);
      const stallCls = worst > 5 ? "bad" : worst > 1 ? "warn" : "good";
      v.stats.innerHTML = `
        <div class="stat ${stallCls}"><span>longest stall</span><b>${worst.toFixed(1)} s</b><small>${cur > 1 ? `now: ${cur.toFixed(1)} s and counting` : `at ${e.stallAt.toFixed(0)} s`}</small></div>
        <div class="stat"><span>agent tok/s (2 s)</span><b>${e.rateOver(2).toFixed(0)}</b><small>all streams</small></div>
        <div class="stat"><span>cold TTFT mean</span><b>${ttft.toFixed(1)} s</b><small>${served.length}/${cold.filter((r) => r.arrival <= simTime).length} served</small></div>
        <div class="stat"><span>KV pool</span><b>${(100 * e.pool.usage()).toFixed(0)}%</b><small>${e.pool.evictions} evictions, ${e.pool.recomputes} recomputes</small></div>
        <div class="stat"><span>GPU</span><b>${busy > 0 ? ((100 * e.busyDecode) / busy).toFixed(0) : 0}% decode</b><small>${e.running.length} running, ${e.waiting.filter((r) => r.arrival <= simTime).length} queued</small></div>
        ${e.bal ? `<div class="stat"><span>balance</span><b>${e.bal.debt >= 0 ? "+" : ""}${e.bal.debt.toFixed(2)} s</b><small>prefill ahead of decode</small></div>` : `<div class="stat"><span>balance</span><b>-</b><small>prefill always first</small></div>`}
      `;
      const items: string[] = [];
      if (e.chunked) items.push(`<li class="chunked">chunking ${e.chunked.tag || "conv " + e.chunked.conv}: ${e.chunked.prefix.toLocaleString()} / ${e.chunked.input.toLocaleString()} tokens</li>`);
      for (const r of e.waiting.filter((r) => r.arrival <= simTime).slice(0, 8)) {
        items.push(`<li>${r.kind === "cold" ? "cold" : "turn"} ${r.tag || "conv " + r.conv}: ${r.input.toLocaleString()} tokens, waiting ${(simTime - r.arrival).toFixed(1)} s</li>`);
      }
      v.queue.innerHTML = items.length ? items.join("") : "<li class='muted'>queue empty</li>";
      const rows: string[] = [];
      let si = 0;
      const palette = ["#22c55e", "#06b6d4", "#eab308", "#f472b6", "#a3e635", "#fb923c", "#c084fc", "#2dd4bf"];
      for (const s of v.side.feed.streams.values()) {
        const rate = e.rateOver(2, s.conv);
        const active = e.running.find((r) => r.conv === s.conv);
        const wait = e.waiting.find((r) => r.conv === s.conv && r.arrival <= simTime);
        const state = active ? (active.prefix < active.input ? "prefilling" : "decoding") : wait ? "queued" : "thinking";
        rows.push(`<li><i style="background:${palette[si++ % palette.length]}"></i>${s.label} <span class="muted">${(s.ctx / 1000).toFixed(0)}k ctx</span> <b>${rate.toFixed(0)} tok/s</b> <span class="state ${state}">${state}</span></li>`);
      }
      v.streams.innerHTML = rows.join("");
    }
  }

  function inject(kind: "cold" | "follow" | "agent"): void {
    const at = simTime;
    const size = num("injectSize") * 1000;
    for (const v of views) {
      if (kind === "cold") v.side.feed.injectCold(v.side.engine, at, size, 300, `cold ${(size / 1000).toFixed(0)}k`);
      else if (kind === "agent") v.side.feed.injectNewAgent(v.side.engine, at, size);
      else {
        const convs = [...v.side.feed.streams.keys()];
        const conv = convs[Math.floor(at * 7) % convs.length];
        v.side.feed.injectFollowUp(v.side.engine, at, conv);
      }
    }
    render();
  }

  window.addEventListener("DOMContentLoaded", () => {
    el("reset").addEventListener("click", reset);
    el("play").addEventListener("click", () => {
      running = !running;
      lastFrame = 0;
      el("play").textContent = running ? "Pause" : "Play";
    });
    el("stepBtn").addEventListener("click", () => {
      step(1);
      render();
    });
    el<HTMLInputElement>("speed").addEventListener("input", () => {
      speed = num("speed");
      el("speedLabel").textContent = `${speed}x`;
    });
    el("injectCold").addEventListener("click", () => inject("cold"));
    el("injectFollow").addEventListener("click", () => inject("follow"));
    el("injectAgent").addEventListener("click", () => inject("agent"));
    for (const id of ["seed", "agents", "ctxMin", "ctxMax", "thinkMin", "thinkMax", "coldEvery", "coldMin", "coldMax", "chunk", "pool", "maxRunning", "hicache", "mixed", "cede"]) {
      el(id).addEventListener("change", reset);
    }
    reset();
    requestAnimationFrame(frame);
  });
}

// ---------------------------------------------------------------- exports

// The tests load sim.js with require; the page gets none of this.
if (typeof module !== "undefined" && module.exports) {
  module.exports = { LOG_CALIBRATION, prefillSeconds, decodeSeconds, Balancer, Pool, Engine, Feed, Side, rng };
}
