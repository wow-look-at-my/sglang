import { test } from "node:test";
import assert from "node:assert/strict";
import { loadPage } from "./dom.mjs";

test("the page builds both sides, animates, injects and resets", () => {
  const page = loadPage();
  page.ready();
  const { el } = page;
  assert.equal(el("clock").textContent, "0.0 s");
  assert.match(el("old-stats").innerHTML, /prefill always first/);
  assert.match(el("new-stats").innerHTML, /prefill ahead of decode/);
  assert.match(el("old-queue").innerHTML, /queue empty|waiting/);
  assert.ok(el("old-timeline").draws.length > 0 && el("new-rates").draws.length > 0);
  assert.equal(page.frames.length, 1);

  // Paused frames only re-arm; Play makes them advance the clock.
  page.frame(0);
  assert.equal(el("clock").textContent, "0.0 s");
  el("play").fire("click");
  assert.equal(el("play").textContent, "Pause");
  page.frame(1000);
  page.frame(1500);
  assert.notEqual(el("clock").textContent, "0.0 s");
  // A long gap between frames is capped, so the sim never leaps.
  page.frame(90000);
  assert.ok(parseFloat(el("clock").textContent) < 60);
  el("play").fire("click");
  assert.equal(el("play").textContent, "Play");

  // Speed slider, single steps and every injector.
  el("speed").value = "30";
  el("speed").fire("input");
  assert.equal(el("speedLabel").textContent, "30x");
  for (let i = 0; i < 70; i++) el("stepBtn").fire("click");
  assert.match(el("old-streams").innerHTML, /agent 1/);
  assert.match(el("old-streams").innerHTML, /(decoding|prefilling|queued|thinking)/);
  el("injectCold").fire("click");
  assert.ok(page.els.get("old-queue").innerHTML.includes("cold 400k") || el("old-stats").innerHTML.includes("served"));
  el("injectAgent").fire("click");
  el("injectFollow").fire("click");
  for (let i = 0; i < 30; i++) el("stepBtn").fire("click");
  assert.match(el("old-stats").innerHTML, /longest stall/);
  assert.match(el("new-streams").innerHTML, /agent 6/);

  // Changing a control rebuilds both sides; Reset zeroes the clock.
  el("hicache").checked = false;
  el("mixed").checked = false;
  el("cede").checked = false;
  el("agents").value = "2";
  el("coldEvery").value = "0";
  el("agents").fire("change");
  assert.equal(el("clock").textContent, "0.0 s");
  el("stepBtn").fire("click");
  el("reset").fire("click");
  assert.equal(el("clock").textContent, "0.0 s");
});

test("a page missing an element fails loudly", () => {
  const page = loadPage();
  page.els.delete("clock");
  assert.throws(() => page.ready(), /missing #clock/);
});
