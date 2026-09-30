import { test } from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import path from "node:path";
import { require, webDir } from "./dom.mjs";

const sim = require(path.join(webDir, "sim.js"));
const { LOG_CALIBRATION: CAL, prefillSeconds, decodeSeconds, Balancer, Pool, Engine, Feed, Side, rng } = sim;

const traffic = {
  seed: 7, agents: 5, ctxMin: 100000, ctxMax: 250000, newMin: 200, newMax: 1400, outMin: 200, outMax: 800,
  thinkMin: 1, thinkMax: 5, coldEvery: 90, coldMin: 150000, coldMax: 400000, shared: 12288,
};

test("cost model prices batches the way the calibration says", () => {
  assert.equal(prefillSeconds(CAL, []), 0);
  const one = prefillSeconds(CAL, [{ tokens: 4096, midCtx: 0 }]);
  assert.ok(Math.abs(one - (CAL.prefillBase + 4096 * CAL.perToken)) < 1e-9);
  // Context makes a chunk dearer.
  assert.ok(prefillSeconds(CAL, [{ tokens: 4096, midCtx: 400000 }]) > one);
  assert.equal(decodeSeconds(CAL, 0, 0), 0);
  assert.ok(Math.abs(decodeSeconds(CAL, 1, 0) - CAL.decodeBase) < 1e-12);
  assert.ok(decodeSeconds(CAL, 4, 1e6) > decodeSeconds(CAL, 1, 0));
});

test("the generator is seeded and uniform", () => {
  const a = rng(42), b = rng(42), c = rng(43);
  const xs = Array.from({ length: 5 }, () => a());
  assert.deepEqual(xs, Array.from({ length: 5 }, () => b()));
  assert.notDeepEqual(xs, Array.from({ length: 5 }, () => c()));
  for (const x of xs) assert.ok(x >= 0 && x < 1);
});

test("balancer defers prefill while it is ahead and lets one burst through", () => {
  const b = new Balancer(4096, true);
  // Nothing contended: state resets and nothing is deferred.
  assert.equal(b.shouldDeferPrefill(false, true, false), false);
  assert.equal(b.shouldDeferPrefill(true, false, false), false);
  // Fresh: a burst of one chunk runs before decode must catch up.
  assert.equal(b.shouldDeferPrefill(true, true, false), false);
  assert.equal(b.prefillTokenBudget(false), -1);
  b.onLaunch(true, 4096, 0, 0);
  b.onFinish(0.3);
  assert.equal(b.burstUsed, 4096);
  assert.equal(b.prefillTokenBudget(false), 0);
  // Prefill charged 0.3 s against no decode: ahead, so deferred.
  assert.equal(b.shouldDeferPrefill(true, true, false), true);
  assert.ok(b.debt > 0);
  // A chunk continuing a long prompt is bounded at its marginal rate.
  assert.equal(b.shouldDeferPrefill(true, true, true), true);
  b.onLaunch(true, 4096, 2, 0.3);
  b.onFinish(0.9);
  assert.ok(b.prefillTokenBudget(true) > 0);
  // Decode catches up: debt falls and the burst reopens.
  for (let i = 0; i < 20; i++) {
    b.onLaunch(false, 0, 3, 1 + i * 0.1);
    b.onFinish(1.1 + i * 0.1);
  }
  assert.equal(b.shouldDeferPrefill(true, true, false), false);
  assert.ok(b.debt <= 0);
  // With no burst the rule is the plain sign of the debt.
  const plain = new Balancer(0, false);
  assert.equal(plain.prefillTokenBudget(false), -1);
  plain.onLaunch(true, 100, 0, 0);
  plain.onFinish(1);
  assert.equal(plain.shouldDeferPrefill(true, true, false), true);
  // A finish with nothing in flight is ignored.
  plain.onFinish(5);
  assert.equal(plain.unsettledN, 0);
});

test("pool caches whole conversations, evicts LRU and keeps a host copy", () => {
  const p = new Pool(1000, 1000);
  assert.deepEqual(p.lookup(1, 100), { dev: 0, host: 0 });
  // Two conversations run and finish; their contexts stay cached.
  p.admit(1, 300, 0, 0, 0, 0);
  p.admit(2, 400, 0, 0, 0, 0);
  assert.equal(p.free(), 300);
  p.release(1, 300, 300, 1);
  p.release(2, 400, 400, 2);
  assert.deepEqual(p.lookup(1, 300), { dev: 299, host: 0 });
  assert.equal(p.free(), 300);
  assert.ok(Math.abs(p.usage() - 0.7) < 1e-12);
  p.touch(1, 5);
  p.touch(9, 5);
  // Evicting for conversation 3 takes the least recently used other one.
  assert.equal(p.evictFor(3, 100), true);
  assert.equal(p.evictions, 1);
  assert.equal(p.entries.get(2).dev, 0);
  assert.equal(p.entries.get(2).host, 400);
  assert.deepEqual(p.lookup(2, 400), { dev: 0, host: 399 });
  // More than can be freed: false, everything idle evicted.
  assert.equal(p.evictFor(3, 5000), false);
  // A conversation seen before that comes back with under half its prefix
  // cached is a recompute; a host hit is a reload.
  p.admit(2, 400, 50, 0, 399, 6);
  assert.equal(p.reloads, 1);
  assert.equal(p.recomputes, 0);
  p.release(2, 450, 450, 7);
  p.admit(2, 450, 10, 0, 0, 8);
  assert.equal(p.recomputes, 1);
  assert.equal(p.recomputeTokens, 450);
  // Without a host tier an evicted entry is gone.
  const q = new Pool(1000, 0);
  q.admit(1, 300, 0, 0, 0, 0);
  q.release(1, 300, 300, 1);
  q.evictFor(2, 100);
  assert.equal(q.entries.has(1), false);
  // A host tier too small to take the entry keeps an earlier host copy.
  const r = new Pool(1000, 100);
  r.admit(1, 300, 0, 0, 0, 0);
  r.release(1, 300, 300, 1);
  r.evictFor(2, 100);
  assert.equal(r.entries.has(1), false);
});

test("old rule stalls decode behind a cold prompt, the balancer does not", () => {
  const sides = {};
  for (const mode of ["old", "new"]) {
    const s = new Side(mode, traffic, CAL, 16, mode === "new" ? 4 : 0, mode === "new", mode === "new");
    s.engine.advanceTo(200);
    sides[mode] = s.engine;
  }
  const old = sides.old, nu = sides.new;
  assert.ok(old.longestStall > 5, `old stall ${old.longestStall}`);
  assert.ok(nu.longestStall < old.longestStall, `new ${nu.longestStall} vs old ${old.longestStall}`);
  assert.ok(old.batches.length > 100 && nu.batches.length > 100);
  assert.ok(nu.batches.some((b) => b.kind === "mixed"), "mixed chunks on the new side");
  assert.ok(old.batches.every((b) => b.kind !== "mixed"));
  assert.ok(old.rateOver(200) > 0 && nu.rateOver(200) > 0);
  assert.ok(nu.rateOver(2, 1) >= 0);
  assert.ok(old.currentStall() >= 0);
  assert.ok(old.busyPrefill > 0 && old.busyDecode > 0);
  assert.ok(nu.bal !== null && old.bal === null);
  // Both sides saw the same traffic.
  assert.equal(old.requests.filter((r) => r.kind === "cold").length, nu.requests.filter((r) => r.kind === "cold").length);
});

test("cede lets short turns ride inside a long prompt's chunks", () => {
  const quiet = { ...traffic, agents: 2, coldEvery: 0 };
  const run = (cede) => {
    const s = new Side("new", quiet, CAL, 16, 0, false, cede);
    s.feed.injectCold(s.engine, 0.5, 300000, 100, "big");
    s.feed.injectFollowUp(s.engine, 0.6, 1);
    s.feed.injectNewAgent(s.engine, 0.7, 2000);
    s.engine.advanceTo(40);
    return s.engine;
  };
  const withCede = run(true), without = run(false);
  const firstTurn = (e) => e.requests.filter((r) => r.kind === "agent" && r.arrival >= 0.6).map((r) => r.firstTok);
  assert.ok(Math.min(...firstTurn(withCede)) <= Math.min(...firstTurn(without)));
  assert.ok(withCede.batches.some((b) => b.cold && b.tokens > 0));
  assert.ok(withCede.requests.find((r) => r.tag === "new conversation"));
});

test("a small pool evicts and recomputes; a host tier reloads instead", () => {
  const small = { ...CAL, poolTokens: 300000 };
  const t = { ...traffic, agents: 6, coldEvery: 30, coldMin: 100000, coldMax: 150000 };
  const noHost = new Side("old", t, small, 16, 0, false, false);
  noHost.engine.advanceTo(150);
  assert.ok(noHost.engine.pool.evictions > 0);
  assert.ok(noHost.engine.pool.recomputes > 0);
  const host = new Side("new", t, small, 16, 4, true, true);
  host.engine.advanceTo(150);
  assert.ok(host.engine.pool.reloads > 0, "host hits reloaded");
});

test("feed and engine edge cases", () => {
  const f = new Feed({ ...traffic, coldEvery: 0, agents: 1 });
  assert.equal(f.coldSchedule.length, 0);
  const e = new Engine({ mode: "old", cal: CAL, maxRunning: 1, hostMul: 0, mixedChunk: false, cede: false }, f);
  f.seed(e);
  // Idle time jumps to the next arrival, and the clock never runs past t.
  e.advanceTo(0.1);
  assert.ok(e.now <= 0.1 + 1);
  // A follow-up on an unknown conversation and a finish for one are ignored.
  f.injectFollowUp(e, 1, 999);
  f.onFinish(e, { conv: 999 }, 1);
  assert.equal(e.requests.length, 1);
  // maxRunning 1 keeps a second arrival waiting while the first runs.
  f.injectCold(e, 0.2, 50000, 50, "a");
  f.injectCold(e, 0.2, 50000, 50, "b");
  e.advanceTo(0.5);
  assert.ok(e.running.length <= 1);
  // A request larger than the pool can never be admitted and stays waiting.
  const tiny = new Engine({ mode: "new", cal: { ...CAL, poolTokens: 1000 }, maxRunning: 4, hostMul: 0, mixedChunk: true, cede: true }, new Feed({ ...traffic, agents: 0, coldEvery: 0 }));
  tiny.add("cold", 1, 0, 5000, 10, "huge");
  tiny.advanceTo(2);
  assert.equal(tiny.waiting.length, 1);
  assert.equal(tiny.now, 2);
  // Long runs trim their batch and delivery histories.
  const long = new Side("new", { ...traffic, agents: 8, coldEvery: 0 }, CAL, 16, 0, true, true);
  long.engine.advanceTo(400);
  assert.ok(long.engine.batches.length <= 20000);
  assert.ok(long.engine.deliveries.length <= 200000);
});

test("node sim.js prints the headless comparison", () => {
  const out = execFileSync(process.execPath, [path.join(webDir, "sim.js")], { encoding: "utf8" });
  const lines = out.trim().split("\n");
  assert.equal(lines.length, 2);
  assert.match(lines[0], /^old: longest stall [\d.]+ s, agent tok\/s [\d.]+, cold TTFT [\d.]+ s over \d+ prompts, recomputes \d+, batches \d+$/);
  assert.match(lines[1], /^new: /);
  const stall = (l) => parseFloat(l.match(/stall ([\d.]+)/)[1]);
  assert.ok(stall(lines[1]) < stall(lines[0]));
});
