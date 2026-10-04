import importlib.util
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
CI_REGISTER_PATH = REPO_ROOT / "python" / "sglang" / "test" / "ci" / "ci_register.py"
HELPER_PATH = REPO_ROOT / "scripts" / "ci" / "utils" / "prune_ghcr_images.py"

DEFAULT_BRANCH = "master"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


register_cpu_ci = _load_module("ci_register", CI_REGISTER_PATH).register_cpu_ci
register_cpu_ci(est_time=0, suite="base-a-test-cpu")


class TestPruneGhcrImages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helper = _load_module("prune_ghcr_images", HELPER_PATH)

    def version(self, version_id, tags, age_days, branch=None):
        created = self.now - timedelta(days=age_days)
        version = {
            "id": version_id,
            "tags": list(tags),
            "created_at": created.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        if branch:
            version["branch"] = branch
        return version

    def plan(
        self, versions, *, live=(), retired=(), min_age_days=7, master_retention=5
    ):
        return self.helper.plan_deletions(
            versions,
            default_branch=DEFAULT_BRANCH,
            live_branches=set(live),
            retired_branches=set(retired),
            now=self.now,
            min_age_days=min_age_days,
            master_retention=master_retention,
        )

    def setUp(self):
        self.now = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)

    def ids(self, plan_result):
        return sorted(version["id"] for version, _ in plan_result)

    def test_default_branch_keeps_head_and_four_more(self):
        versions = [
            self.version(
                index, [f"dev-{index:07x}", f"dev-{index:07x}-amd64"], age_days
            )
            for index, age_days in enumerate([8, 9, 10, 11, 12, 13, 14])
        ]

        deletions, keeps = self.plan(versions)

        self.assertEqual(self.ids(deletions), [5, 6])
        self.assertEqual(self.ids(keeps), [0, 1, 2, 3, 4])

    def test_default_branch_pointer_does_not_use_up_a_retention_slot(self):
        pointer = self.version(99, ["dev"], age_days=400)
        images = [
            self.version(index, [f"dev-{index:07x}"], age_days)
            for index, age_days in enumerate([8, 9, 10, 11, 12, 13])
        ]

        deletions, keeps = self.plan([pointer, *images])

        self.assertEqual(self.ids(deletions), [5])
        self.assertEqual(self.ids(keeps), [0, 1, 2, 3, 4, 99])

    def test_live_branch_keeps_its_head_image(self):
        head = self.version(1, ["dev-claude-cool-wind"], age_days=90)

        deletions, keeps = self.plan([head], live=["claude/cool-wind"])

        self.assertEqual(deletions, [])
        self.assertEqual(self.ids(keeps), [1])

    def test_branch_head_is_kept_when_the_tag_sanitizes_its_name(self):
        head = self.version(
            1, ["dev-claude-cool-wind"], age_days=90, branch="claude/cool-wind"
        )
        other = self.version(2, ["dev-claude-cool-wind-amd64"], age_days=90)

        deletions, keeps = self.plan([head, other], live=["claude/cool-wind"])

        self.assertEqual(deletions, [])
        self.assertEqual(self.ids(keeps), [1, 2])

    def test_merged_branch_images_are_deleted(self):
        head = self.version(1, ["dev-claude-cool-wind"], age_days=30)

        deletions, keeps = self.plan(
            [head], live=["claude/cool-wind"], retired=["claude/cool-wind"]
        )

        self.assertEqual(self.ids(deletions), [1])
        self.assertEqual(keeps, [])

    def test_deleted_branch_images_are_deleted(self):
        head = self.version(1, ["dev-claude-cool-wind"], age_days=30)

        deletions, keeps = self.plan([head], live=[])

        self.assertEqual(self.ids(deletions), [1])
        self.assertEqual(keeps, [])

    def test_nothing_younger_than_the_min_age_is_deleted(self):
        fresh_head = self.version(1, ["dev-claude-cool-wind"], age_days=6)
        fresh_master = self.version(2, ["dev-aaaaaaa"], age_days=6, branch=None)

        deletions, keeps = self.plan(
            [fresh_head, fresh_master], retired=["claude/cool-wind"]
        )

        self.assertEqual(deletions, [])
        self.assertEqual(self.ids(keeps), [1, 2])

    def test_cache_and_kernel_wheel_tags_are_never_deleted(self):
        cache = self.version(1, ["buildcache-amd64"], age_days=365)
        wheel = self.version(2, ["sgl-kernel-wheel-x86_64-deadbeef"], age_days=365)

        deletions, keeps = self.plan([cache, wheel])

        self.assertEqual(deletions, [])
        self.assertEqual(self.ids(keeps), [1, 2])

    def test_untagged_versions_are_never_deleted(self):
        orphan = self.version(1, [], age_days=365)

        deletions, keeps = self.plan([orphan])

        self.assertEqual(deletions, [])
        self.assertEqual(self.ids(keeps), [1])

    def test_recorded_branch_attributes_an_image_the_tag_cannot_name(self):
        head = self.version(
            1, ["dev-claude-cool-wind-2"], age_days=90, branch="claude/cool-wind-2"
        )

        deletions, keeps = self.plan([head], live=["claude/cool-wind-2"])

        self.assertEqual(deletions, [])
        self.assertEqual(self.ids(keeps), [1])

    def test_arch_companions_follow_their_image(self):
        versions = [
            self.version(1, ["dev-bbbb"], age_days=100),
            self.version(2, ["dev-bbbb-amd64"], age_days=100),
            self.version(3, ["dev-cccc"], age_days=1),
        ]

        deletions, keeps = self.plan(versions, master_retention=1)

        self.assertEqual(self.ids(deletions), [1, 2])
        self.assertEqual(self.ids(keeps), [3])

    def test_tag_sanitize_matches_the_workflow_tr(self):
        self.assertEqual(
            self.helper.tag_sanitize("claude/cool-wind"), "claude-cool-wind"
        )
        self.assertEqual(self.helper.tag_sanitize("fix_thing.v2"), "fix_thing.v2")
        self.assertEqual(self.helper.tag_sanitize("a" * 120), "a" * 100)

    def test_short_sha_detection_distinguishes_commits_from_branches(self):
        self.assertTrue(self.helper.is_short_sha("abcdef1"))
        self.assertFalse(self.helper.is_short_sha("claude"))
        self.assertFalse(self.helper.is_short_sha("abcdef12"))
        self.assertFalse(self.helper.is_short_sha("abcdefg"))


if __name__ == "__main__":
    unittest.main()
