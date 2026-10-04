/**
 * Retention for the sglang images Fork CI publishes to ghcr.io.
 *
 * Every push publishes one image. The default branch tags it `dev-<commit>`
 * and moves `dev`; any other branch tags it `dev-<branch>`. The newest image
 * of a live branch is that branch's HEAD and stays, and the default branch
 * keeps its newest MASTER_RETENTION images. Everything else goes once it is
 * older than MIN_AGE_DAYS: a superseded image of a live branch, every image of
 * a branch that merged or was deleted, and older default-branch images.
 *
 * `planDeletions` reads no registry and writes nothing, so the rules are
 * tested directly; `prune-ghcr-images.ts` does the registry work.
 */

export const ARCH_SUFFIX = "-amd64";
export const POINTER_TAG = "dev";
export const SHORT_SHA_LENGTH = 7;
/** Never delete an image younger than this, whatever the other rules say. */
export const MIN_AGE_DAYS = 7;
/** Default-branch images to keep: the HEAD image plus four more. */
export const MASTER_RETENTION = 5;

export interface ImageVersion {
  id: number;
  /** The manifest digest, as the packages API reports it. */
  name: string;
  tags: string[];
  /** ISO timestamp of when the image was pushed. */
  createdAt: string;
  /** Branch Fork CI recorded on the image, when the caller could read it back. */
  branch?: string;
}

export interface Verdict {
  version: ImageVersion;
  reason: string;
}

export interface Decision {
  deletions: Verdict[];
  keeps: Verdict[];
}

export interface PlanInput {
  versions: ImageVersion[];
  defaultBranch: string;
  /** Branches that still exist on the remote. */
  liveBranches: string[];
  /** Branches whose pull request merged, or that were deleted. */
  retiredBranches: string[];
  now: Date;
  minAgeDays?: number;
  masterRetention?: number;
}

/** The tag without the per-arch suffix Fork CI appends. */
export function baseTag(tag: string): string {
  return tag.endsWith(ARCH_SUFFIX) ? tag.slice(0, -ARCH_SUFFIX.length) : tag;
}

/** Whether Fork CI owns the tag: `dev`, `dev-<commit>` or `dev-<branch>`. */
export function isManagedTag(tag: string): boolean {
  const base = baseTag(tag);
  return base === POINTER_TAG || base.startsWith(`${POINTER_TAG}-`);
}

export function isShortSha(text: string): boolean {
  return text.length === SHORT_SHA_LENGTH && /^[0-9a-f]+$/.test(text);
}

/** The branch name as it appears in a tag, matching fork-ci.yml's `tr`. */
export function tagSanitize(branch: string): string {
  const mapped = branch
    .split("")
    .map((char) => (/^[A-Za-z0-9_.-]$/.test(char) ? char : "-"))
    .join("");
  return mapped.slice(0, 100);
}

/** The commit a default-branch image was built from, or null for another branch. */
export function masterCommit(version: ImageVersion): string | null {
  for (const tag of version.tags) {
    const base = baseTag(tag);
    if (base === POINTER_TAG) continue;
    const suffix = base.slice(POINTER_TAG.length + 1);
    if (isShortSha(suffix)) return suffix;
  }
  return null;
}

/** The branch a managed image belongs to; a `dev-<commit>` tag is the default branch. */
export function versionBranch(version: ImageVersion, defaultBranch: string): string {
  if (version.branch) return version.branch;
  for (const tag of version.tags) {
    if (!isManagedTag(tag)) continue;
    const base = baseTag(tag);
    if (base === POINTER_TAG) return defaultBranch;
    const suffix = base.slice(POINTER_TAG.length + 1);
    return isShortSha(suffix) ? defaultBranch : suffix;
  }
  return defaultBranch;
}

export function ageDays(version: ImageVersion, now: Date): number {
  const created = new Date(version.createdAt);
  return (now.getTime() - created.getTime()) / 86_400_000;
}

export function planDeletions(input: PlanInput): Decision {
  const minAgeDays = input.minAgeDays ?? MIN_AGE_DAYS;
  const masterRetention = input.masterRetention ?? MASTER_RETENTION;
  const live = new Set(input.liveBranches);
  const retired = new Set(input.retiredBranches);
  const liveTagNames = new Set(input.liveBranches.map(tagSanitize));
  const retiredTagNames = new Set(input.retiredBranches.map(tagSanitize));
  const deletions: Verdict[] = [];
  const keeps: Verdict[] = [];
  const master = new Map<string, ImageVersion[]>();
  const tooYoung = (version: ImageVersion) =>
    ageDays(version, input.now) < minAgeDays;

  for (const version of input.versions) {
    const managed = version.tags.filter(isManagedTag);
    if (managed.length !== version.tags.length) {
      keeps.push({ version, reason: "not a Fork CI image tag" });
      continue;
    }
    if (managed.length === 0) {
      keeps.push({ version, reason: "untagged" });
      continue;
    }
    if (managed.some((tag) => baseTag(tag) === POINTER_TAG)) {
      keeps.push({ version, reason: `${input.defaultBranch} pointer` });
      continue;
    }
    const commit = masterCommit(version);
    if (commit) {
      const group = master.get(commit) ?? [];
      group.push(version);
      master.set(commit, group);
      continue;
    }
    const branch = versionBranch(version, input.defaultBranch);
    if (live.has(branch) || (liveTagNames.has(branch) && !retiredTagNames.has(branch))) {
      keeps.push({ version, reason: `HEAD of live branch ${branch}` });
      continue;
    }
    if (tooYoung(version)) {
      keeps.push({ version, reason: `younger than ${minAgeDays} days` });
      continue;
    }
    deletions.push({
      version,
      reason:
        retired.has(branch) || retiredTagNames.has(branch)
          ? `branch ${branch} merged or was deleted`
          : `no branch named ${branch} on the remote`,
    });
  }

  // Ranked over every default-branch image, so the retention count is what
  // bounds the history: an image a week old does not hold a slot that then
  // falls to the newest of the images the age guard has already spared.
  const ranked = [...master.entries()].sort((left, right) => {
    const newest = (group: ImageVersion[]) =>
      Math.min(...group.map((version) => ageDays(version, input.now)));
    return newest(left[1]) - newest(right[1]);
  });
  ranked.forEach(([, images], index) => {
    for (const version of images) {
      if (index < masterRetention) {
        keeps.push({
          version,
          reason: `one of ${masterRetention} latest ${input.defaultBranch} images`,
        });
      } else if (tooYoung(version)) {
        keeps.push({ version, reason: `younger than ${minAgeDays} days` });
      } else {
        deletions.push({ version, reason: "superseded default-branch image" });
      }
    }
  });
  return { deletions, keeps };
}
