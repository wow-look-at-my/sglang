// A stand-in for the page: the elements index.html declares, with the values it gives them, canvases whose 2d context records what was drawn.
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import vm from "node:vm";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
export const webDir = path.join(here, "..");
export const require = createRequire(import.meta.url);

export class FakeElement {
  constructor(id, attrs) {
    this.id = id;
    this.value = attrs.value ?? "";
    this.checked = attrs.checked ?? false;
    this.width = Number(attrs.width ?? 0);
    this.height = Number(attrs.height ?? 0);
    this.textContent = "";
    this.innerHTML = "";
    this.handlers = new Map();
    this.draws = [];
  }
  addEventListener(type, fn) {
    if (!this.handlers.has(type)) this.handlers.set(type, []);
    this.handlers.get(type).push(fn);
  }
  fire(type) {
    for (const fn of this.handlers.get(type) ?? []) fn();
  }
  getContext() {
    const draws = this.draws;
    return new Proxy(
      {},
      {
        get: (_, prop) => (...args) => draws.push([prop, ...args]),
        set: () => true,
      },
    );
  }
}

// Reads every element with an id out of index.html, with the value,
// checked and size attributes the markup gives it.
export function elementsFromIndex() {
  const html = readFileSync(path.join(webDir, "index.html"), "utf8");
  const els = new Map();
  for (const m of html.matchAll(/<(\w+)([^>]*\bid="([^"]+)"[^>]*)>/g)) {
    const attrs = {};
    for (const a of m[2].matchAll(/(\w+)="([^"]*)"/g)) attrs[a[1]] = a[2];
    if (/\schecked\b/.test(m[2])) attrs.checked = true;
    els.set(m[3], new FakeElement(m[3], attrs));
  }
  return els;
}

// Loads sim.js as the page would, against the fake document. Returns the
// elements, the DOMContentLoaded and animation-frame hooks, and the console
// output.
export function loadPage() {
  const els = elementsFromIndex();
  const frames = [];
  const listeners = new Map();
  const logs = [];
  const document = { getElementById: (id) => els.get(id) ?? null };
  const window = {
    addEventListener: (type, fn) => listeners.set(type, fn),
  };
  const sandbox = {
    document,
    window,
    requestAnimationFrame: (cb) => frames.push(cb),
    console: { log: (...a) => logs.push(a.join(" ")) },
    Math,
    Map,
    Set,
    Number,
    Error,
    Array,
  };
  const code = readFileSync(path.join(webDir, "sim.js"), "utf8");
  vm.runInNewContext(code, vm.createContext(sandbox), { filename: path.join(webDir, "sim.js") });
  return {
    els,
    el: (id) => els.get(id),
    ready: () => listeners.get("DOMContentLoaded")(),
    // Runs the next queued animation frame at timestamp ts (milliseconds).
    frame: (ts) => {
      const cb = frames.shift();
      cb(ts);
    },
    frames,
    logs,
  };
}
