import assert from "node:assert/strict";
import { test } from "node:test";

import {
  type Decision,
  type ImageVersion,
  isShortSha,
  planDeletions,
  tagSanitize,
} from "./ghcr-image-retention.ts";

const NOW = new Date("2026-10-04T12:00:00Z");
const DEFAULT_BRANCH = "master";

function version(
  id: number,
  tags: string[],
  ageDays: number,
  branch?: string,
): ImageVersion {
  const created = new Date(NOW.getTime() - ageDays * 86_400_000);
  const image: ImageVersion = {
    id,
    name: `sha256:${id}`,
    tags,
    createdAt: created.toISOString().replace(/\.\d{3}Z$/, "Z"),
  };
  if (branch) image.branch = branch;
  return image;
}

function plan(
  versions: ImageVersion[],
  options: {
    live?: string[];
    retired?: string[];
    masterRetention?: number;
  } = {},
): Decision {
  return planDeletions({
    versions,
    defaultBranch: DEFAULT_BRANCH,
    liveBranches: options.live ?? [],
    retiredBranches: options.retired ?? [],
    now: NOW,
    masterRetention: options.masterRetention,
  });
}

function ids(verdicts: Decision["deletions"]): number[] {
  return verdicts.map((verdict) => verdict.version.id).sort((a, b) => a - b);
}

test("the default branch keeps its HEAD image and four more", () => {
  const versions = [8, 9, 10, 11, 12, 13, 14].map((age, index) =>
    version(index, [`dev-${index.toString(16).padStart(7, "0")}`], age),
  );

  const { deletions, keeps } = plan(versions);

  assert.deepEqual(ids(deletions), [5, 6]);
  assert.deepEqual(ids(keeps), [0, 1, 2, 3, 4]);
});

test("the dev pointer does not use up a retention slot", () => {
  const versions = [
    version(99, ["dev"], 400),
    ...[8, 9, 10, 11, 12, 13].map((age, index) =>
      version(index, [`dev-${index.toString(16).padStart(7, "0")}`], age),
    ),
  ];

  const { deletions, keeps } = plan(versions);

  assert.deepEqual(ids(deletions), [5]);
  assert.deepEqual(ids(keeps), [0, 1, 2, 3, 4, 99]);
});

test("a live branch keeps its HEAD image", () => {
  const { deletions, keeps } = plan(
    [version(1, ["dev-claude-cool-wind"], 90)],
    { live: ["claude/cool-wind"] },
  );

  assert.deepEqual(deletions, []);
  assert.deepEqual(ids(keeps), [1]);
});

test("a branch HEAD is kept when its tag sanitizes the branch name", () => {
  const versions = [
    version(1, ["dev-claude-cool-wind"], 90, "claude/cool-wind"),
    version(2, ["dev-claude-cool-wind-amd64"], 90),
  ];

  const { deletions, keeps } = plan(versions, { live: ["claude/cool-wind"] });

  assert.deepEqual(deletions, []);
  assert.deepEqual(ids(keeps), [1, 2]);
});

test("a merged branch's images are deleted", () => {
  const { deletions, keeps } = plan(
    [version(1, ["dev-claude-cool-wind"], 30)],
    { live: ["claude/cool-wind"], retired: ["claude/cool-wind"] },
  );

  assert.deepEqual(ids(deletions), [1]);
  assert.deepEqual(keeps, []);
});

test("a deleted branch's images are deleted", () => {
  const { deletions, keeps } = plan([version(1, ["dev-claude-cool-wind"], 30)]);

  assert.deepEqual(ids(deletions), [1]);
  assert.deepEqual(keeps, []);
});

test("nothing younger than the minimum age is deleted", () => {
  const versions = [
    version(1, ["dev-claude-cool-wind"], 6),
    version(2, ["dev-aaaaaaa"], 6),
  ];

  const { deletions, keeps } = plan(versions, { retired: ["claude/cool-wind"] });

  assert.deepEqual(deletions, []);
  assert.deepEqual(ids(keeps), [1, 2]);
});

test("cache and kernel wheel tags are never deleted", () => {
  const versions = [
    version(1, ["buildcache-amd64"], 365),
    version(2, ["sgl-kernel-wheel-x86_64-deadbeef"], 365),
  ];

  const { deletions, keeps } = plan(versions);

  assert.deepEqual(deletions, []);
  assert.deepEqual(ids(keeps), [1, 2]);
});

test("untagged images are never deleted", () => {
  const { deletions, keeps } = plan([version(1, [], 365)]);

  assert.deepEqual(deletions, []);
  assert.deepEqual(ids(keeps), [1]);
});

test("a recorded branch attributes an image its tag cannot name", () => {
  const versions = [
    version(1, ["dev-claude-cool-wind-2"], 90, "claude/cool-wind-2"),
  ];

  const { deletions, keeps } = plan(versions, { live: ["claude/cool-wind-2"] });

  assert.deepEqual(deletions, []);
  assert.deepEqual(ids(keeps), [1]);
});

test("per-arch companions follow their image", () => {
  const versions = [
    version(1, ["dev-bbbbbbb"], 100),
    version(2, ["dev-bbbbbbb-amd64"], 100),
    version(3, ["dev-ccccccc"], 8),
  ];

  const { deletions, keeps } = plan(versions, { masterRetention: 1 });

  assert.deepEqual(ids(deletions), [1, 2]);
  assert.deepEqual(ids(keeps), [3]);
});

test("tag sanitizing matches the workflow's tr", () => {
  assert.equal(tagSanitize("claude/cool-wind"), "claude-cool-wind");
  assert.equal(tagSanitize("fix_thing.v2"), "fix_thing.v2");
  assert.equal(tagSanitize("a".repeat(120)).length, 100);
});

test("a short sha tells a commit apart from a branch", () => {
  assert.equal(isShortSha("abcdef1"), true);
  assert.equal(isShortSha("claude"), false);
  assert.equal(isShortSha("abcdef12"), false);
  assert.equal(isShortSha("abcdefg"), false);
});
