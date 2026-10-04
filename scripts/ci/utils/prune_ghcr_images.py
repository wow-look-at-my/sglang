"""Retention for the sglang images Fork CI publishes to ghcr.io.

Every branch push publishes one image, tagged `dev-<branch>` (plus its
`dev-<branch>-amd64` companion); the default branch publishes `dev` and
`dev-<commit>`. The newest image of a live branch is its HEAD and stays; older
images of that branch, images of branches that are gone or merged, and
superseded default-branch images are removable once they are older than
MIN_AGE_DAYS.

The rules live in `plan_deletions`, which reads no registry and writes
nothing, so they are tested directly.
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import urllib.parse
import urllib.request

ARCH_SUFFIX = "-amd64"
POINTER_TAG = "dev"
SHORT_SHA_LENGTH = 7
MIN_AGE_DAYS = 7
MASTER_RETENTION = 5
GHCR = "https://ghcr.io"


def base_tag(tag: str) -> str:
    """The tag without the per-arch suffix Fork CI appends."""
    return tag[: -len(ARCH_SUFFIX)] if tag.endswith(ARCH_SUFFIX) else tag


def is_managed_tag(tag: str) -> bool:
    """Whether Fork CI owns the tag: `dev`, `dev-<commit>` or `dev-<branch>`."""
    base = base_tag(tag)
    return base == POINTER_TAG or base.startswith(POINTER_TAG + "-")


def is_short_sha(text: str) -> bool:
    return len(text) == SHORT_SHA_LENGTH and all(
        char in "0123456789abcdef" for char in text
    )


def tag_sanitize(branch: str) -> str:
    """The branch name as it appears in a tag, matching fork-ci.yml's `tr`."""
    return "".join(
        char if char.isascii() and (char.isalnum() or char in "_.-") else "-"
        for char in branch
    )[:100]


def version_branch(version: dict, default_branch: str) -> str:
    """The branch a managed image belongs to; `dev-<commit>` tags are the default branch."""
    if version.get("branch"):
        return version["branch"]
    for tag in version.get("tags") or ():
        if not is_managed_tag(tag):
            continue
        base = base_tag(tag)
        if base == POINTER_TAG:
            return default_branch
        suffix = base[len(POINTER_TAG) + 1 :]
        return default_branch if is_short_sha(suffix) else suffix
    return default_branch


def parse_created_at(version: dict) -> datetime.datetime:
    stamp = version["created_at"].replace("Z", "+00:00")
    parsed = datetime.datetime.fromisoformat(stamp)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed


def age_days(version: dict, now: datetime.datetime) -> float:
    return (now - parse_created_at(version)).total_seconds() / 86400


def plan_deletions(
    versions: list[dict],
    *,
    default_branch: str,
    live_branches: set[str],
    retired_branches: set[str],
    now: datetime.datetime,
    min_age_days: int = MIN_AGE_DAYS,
    master_retention: int = MASTER_RETENTION,
) -> tuple[list[tuple[dict, str]], list[tuple[dict, str]]]:
    """Split versions into (deletions, keeps) as (version, reason) pairs.

    A version is an image the registry stores: its `tags`, `created_at`, and
    optionally the branch Fork CI recorded on it. `live_branches` still exist on
    the remote and `retired_branches` merged or were deleted, so an image of a
    branch in neither set is attributed from its tag alone.
    """
    keeps: list[tuple[dict, str]] = []
    deletions: list[tuple[dict, str]] = []
    master: dict[str, list[dict]] = {}
    live_tag_names = {tag_sanitize(branch) for branch in live_branches}
    retired_tag_names = {tag_sanitize(branch) for branch in retired_branches}

    for version in versions:
        tags = list(version.get("tags") or ())
        managed = [tag for tag in tags if is_managed_tag(tag)]
        if len(managed) != len(tags):
            keeps.append((version, "not a Fork CI image tag"))
            continue
        if not managed:
            keeps.append((version, "untagged"))
            continue
        if age_days(version, now) < min_age_days:
            keeps.append((version, f"younger than {min_age_days} days"))
            continue
        if any(base_tag(tag) == POINTER_TAG for tag in managed):
            keeps.append((version, f"{default_branch} pointer"))
            continue
        commit = _master_key(version)
        if commit:
            master.setdefault(commit, []).append(version)
            continue
        branch = version_branch(version, default_branch)
        if branch in live_branches or (
            branch in live_tag_names and branch not in retired_tag_names
        ):
            keeps.append((version, f"HEAD of live branch {branch}"))
        elif branch in retired_branches or branch in retired_tag_names:
            deletions.append((version, f"branch {branch} merged or was deleted"))
        else:
            deletions.append((version, f"no branch named {branch} on the remote"))

    ranked = sorted(
        master.items(),
        key=lambda item: min(age_days(version, now) for version in item[1]),
    )
    for index, (_, images) in enumerate(ranked):
        for version in images:
            if index < master_retention:
                keeps.append(
                    (
                        version,
                        f"one of {master_retention} latest {default_branch} images",
                    )
                )
            else:
                deletions.append((version, "superseded default-branch image"))
    return deletions, keeps


def _master_key(version: dict) -> str | None:
    """The commit a default-branch image was built from, or None for another branch."""
    for tag in version.get("tags") or ():
        base = base_tag(tag)
        if base == POINTER_TAG:
            continue
        suffix = base[len(POINTER_TAG) + 1 :]
        if is_short_sha(suffix):
            return suffix
    return None


def parse_versions(pages: list[list[dict]]) -> list[dict]:
    return [version for page in pages for version in page]


def gh_api(*args: str, expect_json: bool = True):
    result = subprocess.run(
        ["gh", "api", *args], check=True, capture_output=True, text=True
    )
    return json.loads(result.stdout) if expect_json else result.stdout


def package_versions(owner: str, package: str) -> list[dict]:
    for scope in ("orgs", "users"):
        try:
            pages = gh_api(
                "--paginate",
                "--slurp",
                f"/{scope}/{owner}/packages/container/{package}/versions",
            )
        except subprocess.CalledProcessError as error:
            last_error = error
            continue
        return parse_versions(pages)
    raise last_error


def remote_branches(repo: str) -> set[str]:
    pages = gh_api("--paginate", "--slurp", f"/repos/{repo}/branches")
    return {branch["name"] for page in pages for branch in page}


def merged_branches(repo: str) -> set[str]:
    pages = gh_api(
        "--paginate", "--slurp", f"/repos/{repo}/pulls?state=closed&per_page=100"
    )
    return {
        pull["head"]["ref"] for page in pages for pull in page if pull.get("merged_at")
    }


def registry_token(repo: str) -> str:
    query = urllib.parse.urlencode(
        {"scope": f"repository:{repo}:pull", "service": "ghcr.io"}
    )
    with urllib.request.urlopen(f"{GHCR}/token?{query}", timeout=30) as response:
        return json.load(response)["token"]


def _registry_get(url: str, token: str, accept: str):
    request = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {token}", "Accept": accept}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def branch_label(repo: str, tag: str, token: str) -> str | None:
    """The branch Fork CI recorded on the image a tag points at."""
    index_accept = ", ".join(
        [
            "application/vnd.oci.image.index.v1+json",
            "application/vnd.docker.distribution.manifest.list.v2+json",
            "application/vnd.oci.image.manifest.v1+json",
            "application/vnd.docker.distribution.manifest.v2+json",
        ]
    )
    manifest = _registry_get(f"{GHCR}/v2/{repo}/manifests/{tag}", token, index_accept)
    children = manifest.get("manifests")
    if children:
        digest = children[0]["digest"]
        manifest = _registry_get(
            f"{GHCR}/v2/{repo}/manifests/{digest}", token, index_accept
        )
    config = manifest.get("config", {}).get("digest")
    if not config:
        return None
    blob = _registry_get(
        f"{GHCR}/v2/{repo}/blobs/{config}",
        token,
        "application/vnd.oci.image.config.v1+json",
    )
    return (blob.get("config", {}).get("Labels") or {}).get("ai.sglang.build.branch")


def attach_branch_labels(
    versions: list[dict], repo: str, live_tag_names: set[str]
) -> None:
    """Fill in `branch` from the image itself, for images whose tag cannot name a live branch."""
    token = None
    for version in versions:
        tags = list(version.get("tags") or ())
        managed = [tag for tag in tags if is_managed_tag(tag)]
        if not managed or len(managed) != len(tags):
            continue
        base = base_tag(managed[0])
        if base == POINTER_TAG or is_short_sha(base[len(POINTER_TAG) + 1 :]):
            continue
        if base[len(POINTER_TAG) + 1 :] in live_tag_names:
            continue
        if token is None:
            token = registry_token(repo)
        try:
            label = branch_label(repo, managed[0], token)
        except Exception as error:  # the tag name stays the fallback
            print(f"could not read the branch label of {managed[0]}: {error}")
            continue
        if label:
            version["branch"] = label


def delete_version(owner: str, package: str, version_id: int) -> None:
    gh_api(
        "-X",
        "DELETE",
        f"/orgs/{owner}/packages/container/{package}/versions/{version_id}",
        expect_json=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete sglang images Fork CI no longer keeps."
    )
    parser.add_argument("--owner", required=True, help="Package owner, e.g. an org.")
    parser.add_argument("--package", default="sglang", help="Container package name.")
    parser.add_argument("--repository", required=True, help="owner/name of the repo.")
    parser.add_argument(
        "--default-branch", default="master", help="Branch whose images keep history."
    )
    parser.add_argument(
        "--deleted-ref",
        default="",
        help="Branch a `delete` event removed, e.g. claude/foo.",
    )
    parser.add_argument(
        "--min-age-days", type=int, default=MIN_AGE_DAYS, help="Never delete younger."
    )
    parser.add_argument(
        "--master-retention",
        type=int,
        default=MASTER_RETENTION,
        help="Default-branch images to keep, the HEAD plus 4 more.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    now = datetime.datetime.now(datetime.timezone.utc)
    versions = package_versions(args.owner, args.package)
    live = remote_branches(args.repository)
    retired = merged_branches(args.repository)
    deleted_ref = args.deleted_ref
    if deleted_ref.startswith("refs/tags/"):
        deleted_ref = ""
    else:
        deleted_ref = deleted_ref.removeprefix("refs/heads/")
    if deleted_ref and deleted_ref != args.default_branch:
        retired.add(deleted_ref)
        live.discard(deleted_ref)
    attach_branch_labels(
        versions, f"{args.owner}/{args.package}", {tag_sanitize(b) for b in live}
    )
    deletions, keeps = plan_deletions(
        versions,
        default_branch=args.default_branch,
        live_branches=live,
        retired_branches=retired,
        now=now,
        min_age_days=args.min_age_days,
        master_retention=args.master_retention,
    )
    for version, reason in deletions:
        tags = ",".join(version.get("tags") or ()) or version.get("name", "?")
        if args.dry_run:
            print(f"would delete {tags} (created {version['created_at']}): {reason}")
            continue
        delete_version(args.owner, args.package, version["id"])
        print(f"deleted {tags} (created {version['created_at']}): {reason}")
    print(f"{len(deletions)} deleted, {len(keeps)} kept, {len(versions)} versions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
