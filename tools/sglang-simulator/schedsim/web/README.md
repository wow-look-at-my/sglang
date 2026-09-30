# Visual scheduler simulator

Open `index.html` in a browser. It needs no server: `sim.js` is the compiled
`sim.ts`, committed beside it, and the page loads it as a plain script.

Two schedulers run side by side on the same traffic: the upstream
prefill-priority rule on the left and this fork's time-share balancer on the
right (with mixed chunk, the cede budget and a host tier, each switchable).
The cost model is schedsim's calibration on the Qwen3.8-Flash-Next log
(`prefill 10 ms + 65.9 us/token + context terms; decode 13.3 ms + 1.5 ms/req;
2.78 accepted tokens per step; 1.4M-token pool`), and the balancer is a port
of `prefill_decode_balancer.py` by way of `internal/sim/balancer.go`.

Traffic: a few agent conversations taking turns (200–1400 new tokens, 200–800
out, 1–5 s think), random cold prompts of 150k–400k tokens on a Poisson
clock, and three buttons that inject a cold prompt, a new conversation, or a
follow-up turn into both sides at the same instant. Each stream draws from its
own seeded generator, so both sides see the same turn sizes and think times
however differently they schedule them.

What to watch: press Play, then inject a cold prompt while the agents are
decoding. On the left the decode steps stop until every chunk of the prompt
has run and the stall counter climbs; on the right decode keeps going, the
prompt finishes later, and the balance stat shows prefill's lead in GPU
seconds. Push the random cold cadence up to see the overload regime, where
the balancer protects running streams at the cost of cold time-to-first-token
and queued turns.

Not modelled: the eviction throttle; the cede here is the simple half-chunk
rule rather than the fork's amortised one; the pool evicts whole
conversations where the server evicts pages.

Rebuild after editing `sim.ts`:

```
npm install     # typescript and c8, once
npm run check   # tsc, then a headless 600 s comparison printed by node
npm test        # node:test suites under c8, 95% line coverage required
```

The tests in `test/` load `sim.js` twice: as a module, for the cost model,
balancer, pool, engine and traffic feed, and as the page would, against a
fake document built from `index.html`'s own elements, for the controls,
the injectors and the drawing. Fork CI runs all of it and fails if the
committed `sim.js` is stale.
