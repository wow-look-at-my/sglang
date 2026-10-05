/** Delete the sglang images Fork CI no longer keeps. */

/** An image as the packages API describes one. */
interface ImageVersion {
  id: number;
  name: string;
  tags: string[];
  createdAt: string;
  branch?: string;
}

interface Verdict {
  version: ImageVersion;
  reason: string;
}

/** The rules module, which is loaded by path rather than imported. */
interface RetentionRules {
  planDeletions: (input: {
    versions: ImageVersion[];
    defaultBranch: string;
    liveBranches: string[];
    retiredBranches: string[];
    now: Date;
    reachableDigests?: ReadonlySet<string>;
  }) => { deletions: Verdict[]; keeps: Verdict[] };
  isManagedTag: (tag: string) => boolean;
  baseTag: (tag: string) => string;
  masterCommit: (version: ImageVersion) => string | null;
  tagSanitize: (branch: string) => string;
  POINTER_TAG: string;
}

// The action type-checks the script as one virtual file beside its own
// globals.
const rules: RetentionRules = require(
  path.join(__dirname, "ghcr-image-retention.ts"),
);
const { POINTER_TAG, baseTag, isManagedTag, masterCommit, planDeletions, tagSanitize } =
  rules;

const PACKAGE_NAME = "sglang";
const REGISTRY = "https://ghcr.io";
const MANIFEST_ACCEPT = [
  "application/vnd.oci.image.index.v1+json",
  "application/vnd.docker.distribution.manifest.list.v2+json",
  "application/vnd.oci.image.manifest.v1+json",
  "application/vnd.docker.distribution.manifest.v2+json",
].join(", ");

/** One image as the packages API lists it: tags live under metadata.container. */
function toImageVersion(raw: {
  id: number;
  name: string;
  created_at: string;
  metadata?: unknown;
}): ImageVersion {
  const container = (raw.metadata as { container?: { tags?: string[] } } | undefined)
    ?.container;
  return {
    id: raw.id,
    name: raw.name,
    tags: container?.tags ?? [],
    createdAt: raw.created_at,
  };
}

interface ManifestChild {
  digest: string;
}

interface Manifest {
  manifests?: ManifestChild[];
  config?: { digest?: string };
}

interface ImageConfig {
  config?: { Labels?: Record<string, string> };
}

async function registryGet<T>(url: string, token: string, accept: string): Promise<T> {
  const response = await fetch(url, {
    headers: { authorization: `Bearer ${token}`, accept },
  });
  if (!response.ok) {
    throw new Error(`${response.status} from ${url}`);
  }
  return (await response.json()) as T;
}

async function pullToken(repoPath: string): Promise<string> {
  const query = new URLSearchParams({
    scope: `repository:${repoPath}:pull`,
    service: "ghcr.io",
  });
  const response = await fetch(`${REGISTRY}/token?${query}`);
  if (!response.ok) {
    throw new Error(`${response.status} from the ghcr token endpoint`);
  }
  const body = (await response.json()) as { token: string };
  return body.token;
}

/** The branch Fork CI recorded on the image a tag points at. */
async function recordedBranch(
  repoPath: string,
  tag: string,
  token: string,
): Promise<string | undefined> {
  const manifest = await registryGet<Manifest>(
    `${REGISTRY}/v2/${repoPath}/manifests/${tag}`,
    token,
    MANIFEST_ACCEPT,
  );
  const child = manifest.manifests?.[0]?.digest;
  const image = child
    ? await registryGet<Manifest>(
        `${REGISTRY}/v2/${repoPath}/manifests/${child}`,
        token,
        MANIFEST_ACCEPT,
      )
    : manifest;
  const digest = image.config?.digest;
  if (!digest) return undefined;
  const config = await registryGet<ImageConfig>(
    `${REGISTRY}/v2/${repoPath}/blobs/${digest}`,
    token,
    "application/vnd.oci.image.config.v1+json",
  );
  return config.config?.Labels?.["ai.sglang.build.branch"];
}

/**
 * Every digest a surviving tag reaches: the tag's own manifest and the
 * manifests below it. An untagged image outside this set cannot be pulled.
 */
async function reachableDigests(
  repoPath: string,
  tags: string[],
  token: string,
): Promise<Set<string>> {
  const reachable = new Set<string>();
  const pending = [...tags];
  const seen = new Set<string>();
  let budget = 500;
  while (pending.length > 0 && budget > 0) {
    const reference = pending.shift() as string;
    if (seen.has(reference)) continue;
    seen.add(reference);
    budget -= 1;
    const response = await fetch(`${REGISTRY}/v2/${repoPath}/manifests/${reference}`, {
      headers: { authorization: `Bearer ${token}`, accept: MANIFEST_ACCEPT },
    });
    if (!response.ok) {
      throw new Error(`${response.status} from the manifest of ${reference}`);
    }
    const digest = response.headers.get("docker-content-digest");
    if (digest) reachable.add(digest);
    const manifest = (await response.json()) as Manifest;
    for (const child of manifest.manifests ?? []) {
      reachable.add(child.digest);
      pending.push(child.digest);
    }
  }
  return reachable;
}

/**
 * Read the recorded branch for images whose tag cannot name a live branch,
 * so an image of a live branch is never taken for an image of a dead one.
 */
async function attachRecordedBranches(
  versions: ImageVersion[],
  repoPath: string,
  liveTagNames: Set<string>,
  getToken: () => Promise<string>,
): Promise<void> {
  for (const version of versions) {
    const managed = version.tags.filter(isManagedTag);
    if (managed.length !== version.tags.length || managed.length === 0) continue;
    const base = baseTag(managed[0]);
    if (base === POINTER_TAG || masterCommit(version)) continue;
    if (liveTagNames.has(base.slice(POINTER_TAG.length + 1))) continue;
    try {
      const branch = await recordedBranch(repoPath, managed[0], await getToken());
      if (branch) version.branch = branch;
    } catch (error) {
      core.warning(`could not read the branch of ${managed[0]}: ${error}`);
    }
  }
}

async function main(): Promise<void> {
  const owner = context.repo.owner;
  const repo = context.repo.repo;
  const repoPath = `${owner}/${PACKAGE_NAME}`;
  const payload = context.payload as {
    inputs?: Record<string, string | boolean>;
    ref?: string;
    repository?: { default_branch?: string };
  };
  const dryRun = String(payload.inputs?.dry_run ?? "") === "true";
  const packageArgs = {
    org: owner,
    package_type: "container" as const,
    package_name: PACKAGE_NAME,
  };

  const rawVersions = await octokit.paginate(
    octokit.rest.packages.getAllPackageVersionsForPackageOwnedByOrg,
    { ...packageArgs, per_page: 100 },
  );
  const versions = rawVersions.map(toImageVersion);

  const branches = await octokit.paginate(octokit.rest.repos.listBranches, {
    owner,
    repo,
    per_page: 100,
  });
  const liveBranches = branches.map((branch) => branch.name);

  const closedPulls = await octokit.paginate(octokit.rest.pulls.list, {
    owner,
    repo,
    state: "closed",
    per_page: 100,
  });
  const retiredBranches = closedPulls
    .filter(
      (pull) =>
        pull.merged_at && pull.head.repo?.full_name === `${owner}/${repo}`,
    )
    .map((pull) => pull.head.ref);
  if (context.eventName === "delete" && String(payload.ref).startsWith("refs/heads/")) {
    retiredBranches.push(String(payload.ref).slice("refs/heads/".length));
  }

  let pullTokenPromise: Promise<string> | undefined;
  const getToken = (): Promise<string> =>
    (pullTokenPromise ??= pullToken(repoPath));
  await attachRecordedBranches(
    versions,
    repoPath,
    new Set(liveBranches.map(tagSanitize)),
    getToken,
  );

  const plan = {
    versions,
    defaultBranch: payload.repository?.default_branch ?? "master",
    liveBranches,
    retiredBranches,
    now: new Date(),
  };
  // A tag that survives is what makes an untagged image reachable, and which
  // tags survive is the first pass's answer.
  const doomedIds = new Set(
    planDeletions(plan).deletions.map(({ version }) => version.id),
  );
  let reachableDigestsSet: ReadonlySet<string> | undefined;
  try {
    reachableDigestsSet = await reachableDigests(
      repoPath,
      versions
        .filter((version) => !doomedIds.has(version.id))
        .flatMap((version) => version.tags),
      await getToken(),
    );
  } catch (error) {
    core.warning(`could not resolve what the surviving tags reference: ${error}`);
  }

  const { deletions, keeps } = planDeletions({
    ...plan,
    reachableDigests: reachableDigestsSet,
  });

  for (const { version, reason } of deletions) {
    if (dryRun) {
      core.info(`would delete ${version.tags.join(",") || version.name}: ${reason}`);
      continue;
    }
    await octokit.rest.packages.deletePackageVersionForOrg({
      ...packageArgs,
      package_version_id: version.id,
    });
    core.info(`deleted ${version.tags.join(",") || version.name}: ${reason}`);
  }
  const counts = new Map<string, number>();
  for (const { reason } of keeps) {
    counts.set(reason, (counts.get(reason) ?? 0) + 1);
  }
  for (const [reason, count] of [...counts].sort()) {
    core.info(`kept ${count}: ${reason}`);
  }
  core.info(
    `${deletions.length} ${dryRun ? "to delete" : "deleted"}, ${keeps.length} kept, ${versions.length} images`,
  );
}

await main();
