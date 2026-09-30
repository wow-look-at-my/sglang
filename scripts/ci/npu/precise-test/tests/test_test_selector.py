"""Tests for the git-diff fallback in scripts/ci/npu/precise-test/test_selector.py.

Scope: ``_diff_from_git`` and the GitHub-API retry / git-fallback block of
``main()``. No network and no real git: ``urllib.request.urlopen``,
``subprocess.run`` and ``time.sleep`` are replaced with fakes.
"""

import json
import subprocess
import urllib.error
from types import SimpleNamespace

import pytest
import test_selector as ts  # module import only: keeps pytest off ts.TestSelector

BASE_SHA = "a" * 40
HEAD_SHA = "b" * 40
FETCH_CMD = ["git", "fetch", "--no-tags", "--depth=1", "origin", BASE_SHA, HEAD_SHA]
DIFF_CMD = ["git", "diff", "--no-color", "--no-ext-diff", BASE_SHA, HEAD_SHA]

# A PR diff that only adds a test file: main() recommends it without needing
# any coverage data or base-content lookups.
PR_DIFF = (
    "diff --git a/test/registered/npu/test_new.py b/test/registered/npu/test_new.py\n"
    "--- /dev/null\n"
    "+++ b/test/registered/npu/test_new.py\n"
    "@@ -0,0 +1 @@\n"
    "+def test_x(): pass\n"
).encode()
NEW_TEST = "test/registered/npu/test_new.py"

PR_URL = "https://api.github.com/repos/o/r/pulls/7"
DIFF_URL = "https://github.com/o/r/pull/7.diff"


# ==================== Fakes ====================


def completed(returncode=0, stdout="", stderr=""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


class FakeGit:
    """subprocess.run stand-in: answers queued results in order, records calls."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


class FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeUrlopen:
    """urlopen stand-in: each call pops the next ``(expected_url, result)``.

    ``result`` is bytes / a dict (JSON-encoded) returned as a response, or an
    exception to raise.
    """

    def __init__(self, *script):
        self.script = list(script)
        self.urls = []

    def __call__(self, req, timeout=None, context=None):
        self.urls.append(req.full_url)
        assert self.script, f"unexpected urlopen({req.full_url})"
        expected_url, result = self.script.pop(0)
        assert req.full_url == expected_url
        if isinstance(result, BaseException):
            raise result
        if isinstance(result, dict):
            result = json.dumps(result).encode()
        return FakeResponse(result)


def pr_ok():
    """The two API responses of one successful attempt."""
    return [
        (PR_URL, {"diff_url": DIFF_URL, "base": {"sha": BASE_SHA}}),
        (DIFF_URL, PR_DIFF),
    ]


def pr_fail(exc=None):
    return (PR_URL, exc or urllib.error.HTTPError(PR_URL, 503, "Unavailable", {}, None))


# ==================== _diff_from_git ====================


class TestDiffFromGit:
    @pytest.mark.parametrize(
        "base,head", [(None, HEAD_SHA), (BASE_SHA, None), ("", HEAD_SHA), (None, None)]
    )
    def test_missing_shas_skip_git(self, tmp_path, monkeypatch, capsys, base, head):
        git = FakeGit()
        monkeypatch.setattr(ts.subprocess, "run", git)
        out = tmp_path / "pr.diff"

        assert ts._diff_from_git(base, head, str(out)) is False
        assert git.calls == []
        assert not out.exists()
        assert "PR_BASE_SHA / PR_HEAD_SHA not set" in capsys.readouterr().out

    def test_fetch_fails(self, tmp_path, monkeypatch, capsys):
        git = FakeGit(completed(128, stderr="  fatal: no such ref \n"))
        monkeypatch.setattr(ts.subprocess, "run", git)
        out = tmp_path / "pr.diff"

        assert ts._diff_from_git(BASE_SHA, HEAD_SHA, str(out)) is False
        assert [cmd for cmd, _ in git.calls] == [FETCH_CMD]  # diff never attempted
        assert git.calls[0][1] == {"capture_output": True, "text": True, "timeout": 300}
        assert not out.exists()
        assert "  git fetch failed: fatal: no such ref\n" in capsys.readouterr().out

    def test_diff_fails(self, tmp_path, monkeypatch, capsys):
        git = FakeGit(
            completed(0), completed(1, stdout=b"", stderr=b"bad \xff object\n")
        )
        monkeypatch.setattr(ts.subprocess, "run", git)
        out = tmp_path / "pr.diff"

        assert ts._diff_from_git(BASE_SHA, HEAD_SHA, str(out)) is False
        assert [cmd for cmd, _ in git.calls] == [FETCH_CMD, DIFF_CMD]
        # diff output is read as bytes (no text=True) so line endings survive
        assert git.calls[1][1] == {"capture_output": True, "timeout": 300}
        assert not out.exists()
        assert "  git diff failed: bad � object\n" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "results",
        [
            [subprocess.TimeoutExpired(FETCH_CMD, 300)],
            [completed(0), FileNotFoundError("git not found")],
        ],
        ids=["fetch-raises", "diff-raises"],
    )
    def test_subprocess_raises(self, tmp_path, monkeypatch, capsys, results):
        monkeypatch.setattr(ts.subprocess, "run", FakeGit(*results))
        out = tmp_path / "pr.diff"

        assert ts._diff_from_git(BASE_SHA, HEAD_SHA, str(out)) is False
        assert not out.exists()
        assert f"  git fallback failed: {results[-1]}" in capsys.readouterr().out

    def test_write_error_is_caught(self, tmp_path, monkeypatch, capsys):
        git = FakeGit(completed(0), completed(0, stdout=b"x", stderr=b""))
        monkeypatch.setattr(ts.subprocess, "run", git)
        missing_dir_file = tmp_path / "no" / "such" / "pr.diff"

        assert ts._diff_from_git(BASE_SHA, HEAD_SHA, str(missing_dir_file)) is False
        assert "git fallback failed:" in capsys.readouterr().out

    def test_success_writes_raw_bytes(self, tmp_path, monkeypatch, capsys):
        payload = b"diff --git a/x b/x\r\n+caf\xc3\xa9\n"
        git = FakeGit(completed(0), completed(0, stdout=payload, stderr=b""))
        monkeypatch.setattr(ts.subprocess, "run", git)
        out = tmp_path / "pr.diff"

        assert ts._diff_from_git("abcdef0123", "9876543210", str(out)) is True
        assert out.read_bytes() == payload
        assert "  Using git diff abcdef0..9876543 from the local checkout\n" in (
            capsys.readouterr().out
        )


# ==================== main(): GitHub API retry + git fallback ====================


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated main(): BASE_DIR, temp dir, env vars, sleep, git and urlopen."""
    base = tmp_path / "base"
    tmp = tmp_path / "tmp"
    base.mkdir()
    tmp.mkdir()
    monkeypatch.setattr(ts, "BASE_DIR", base)
    monkeypatch.setattr(ts.tempfile, "gettempdir", lambda: str(tmp))
    for var in ("GITHUB_TOKEN", "GH_TOKEN", "PR_BASE_SHA", "PR_HEAD_SHA"):
        monkeypatch.delenv(var, raising=False)

    sleeps = []
    monkeypatch.setattr(ts.time, "sleep", sleeps.append)
    git = FakeGit()  # no git call is expected unless a test queues results
    monkeypatch.setattr(ts.subprocess, "run", git)

    map_file = tmp_path / "map.json"
    map_file.write_text("{}")  # empty map: main() loads it instead of building one

    ns = SimpleNamespace(
        sleeps=sleeps,
        git=git,
        diff_file=tmp / "pr.diff",
        output=base / "recommended_pytest_paths.txt",
    )

    def urlopen(*script):
        fake = FakeUrlopen(*script)
        monkeypatch.setattr(ts.urllib.request, "urlopen", fake)
        return fake

    def run():
        argv = ["test_selector.py", "-pr", "o/r#7", "-s", str(tmp_path / "src")]
        monkeypatch.setattr("sys.argv", argv + ["-m", str(map_file)])
        ts.main()

    ns.urlopen = urlopen
    ns.run = run
    return ns


class TestMainRetryAndFallback:
    def test_api_succeeds_first_try(self, env, capsys):
        api = env.urlopen(*pr_ok())

        env.run()
        out = capsys.readouterr().out

        assert api.urls == [PR_URL, DIFF_URL]
        assert env.sleeps == []
        assert env.git.calls == []
        assert env.diff_file.read_bytes() == PR_DIFF
        assert "Attempt 1/5 to get PR diff via GitHub API" in out
        assert "Using GitHub API to get diff" in out
        assert "Attempt 2/5" not in out
        assert "trying the local git checkout" not in out
        assert f"PR diff saved to: {env.diff_file}" in out
        assert env.output.read_text() == NEW_TEST + "\n"

    def test_api_fails_then_succeeds(self, env, capsys):
        api = env.urlopen(
            pr_fail(),
            # Second attempt: PR JSON without diff_url is also a failed attempt
            (PR_URL, {"base": {"sha": BASE_SHA}}),
            *pr_ok(),
        )

        env.run()
        out = capsys.readouterr().out

        assert api.urls == [PR_URL, PR_URL, PR_URL, DIFF_URL]
        assert env.sleeps == [1, 2]  # 2 ** (attempt - 1) after attempts 1 and 2
        assert env.git.calls == []
        assert "Attempt 1 failed: HTTP Error 503: Unavailable" in out
        assert "Attempt 2 failed: Cannot get diff URL" in out
        assert "Attempt 3/5" in out and "Attempt 4/5" not in out
        assert "Using GitHub API to get diff" in out
        assert "trying the local git checkout" not in out
        assert env.output.read_text() == NEW_TEST + "\n"

    def test_last_attempt_succeeds_without_extra_sleep(self, env, capsys):
        env.urlopen(*[pr_fail()] * 4, *pr_ok())

        env.run()
        out = capsys.readouterr().out

        assert env.sleeps == [1, 2, 4, 8]
        assert "Attempt 5/5" in out
        assert "trying the local git checkout" not in out
        assert env.output.read_text() == NEW_TEST + "\n"

    def test_all_attempts_fail_git_fallback_succeeds(self, env, monkeypatch, capsys):
        monkeypatch.setenv("PR_BASE_SHA", BASE_SHA)
        monkeypatch.setenv("PR_HEAD_SHA", HEAD_SHA)
        # The diff download itself failing counts as a failed attempt too
        api = env.urlopen(
            *[pr_fail(urllib.error.URLError("down"))] * 4,
            (PR_URL, {"diff_url": DIFF_URL, "base": {"sha": "stale"}}),
            (DIFF_URL, ConnectionResetError("reset")),
        )
        env.git.results = [completed(0), completed(0, stdout=PR_DIFF, stderr=b"")]

        env.run()
        out = capsys.readouterr().out

        assert api.script == []
        assert len(api.urls) == 6
        assert env.sleeps == [1, 2, 4, 8]  # no sleep after the final attempt
        assert out.count("failed: <urlopen error down>") == 4
        assert "Attempt 5 failed: reset" in out
        assert "All 5 API attempts failed; trying the local git checkout" in out
        assert [cmd for cmd, _ in env.git.calls] == [FETCH_CMD, DIFF_CMD]
        assert "Using git diff aaaaaaa..bbbbbbb from the local checkout" in out
        assert "Could not get the PR diff from git either" not in out
        assert env.diff_file.read_bytes() == PR_DIFF
        assert f"PR diff saved to: {env.diff_file}" in out
        assert env.output.read_text() == NEW_TEST + "\n"

    def test_fallback_sets_base_sha_from_env(self, env, monkeypatch, capsys):
        """base_sha must come from PR_BASE_SHA so base content can be fetched."""
        monkeypatch.setenv("PR_BASE_SHA", BASE_SHA)
        monkeypatch.setenv("PR_HEAD_SHA", HEAD_SHA)
        product_diff = (
            b"--- a/python/sglang/srt/mod.py\n+++ b/python/sglang/srt/mod.py\n"
            b"@@ -2,1 +2,1 @@\n-x = 1\n+x = 2\n"
        )
        contents_url = (
            "https://api.github.com/repos/o/r/contents/python/sglang/srt/mod.py"
            f"?ref={BASE_SHA}"
        )
        api = env.urlopen(
            *[pr_fail()] * 5,
            (contents_url, {"encoding": "base64", "content": "eCA9IDEKeCA9IDEK"}),
        )
        env.git.results = [completed(0), completed(0, stdout=product_diff, stderr=b"")]

        env.run()
        out = capsys.readouterr().out

        assert api.urls[-1] == contents_url
        assert "no base content" not in out
        assert "srt/mod.py: 2" in out

    @pytest.mark.parametrize(
        "env_vars,git_results,message",
        [
            ({}, [], "PR_BASE_SHA / PR_HEAD_SHA not set"),
            ({"PR_BASE_SHA": BASE_SHA}, [], "PR_BASE_SHA / PR_HEAD_SHA not set"),
            (
                {"PR_BASE_SHA": BASE_SHA, "PR_HEAD_SHA": HEAD_SHA},
                [completed(1, stderr="fatal: couldn't find remote ref")],
                "git fetch failed: fatal: couldn't find remote ref",
            ),
            (
                {"PR_BASE_SHA": BASE_SHA, "PR_HEAD_SHA": HEAD_SHA},
                [completed(0), completed(128, stdout=b"", stderr=b"bad revision")],
                "git diff failed: bad revision",
            ),
            (
                {"PR_BASE_SHA": BASE_SHA, "PR_HEAD_SHA": HEAD_SHA},
                [OSError("git missing")],
                "git fallback failed: git missing",
            ),
        ],
        ids=["no-shas", "no-head", "fetch-fails", "diff-fails", "git-raises"],
    )
    def test_all_attempts_fail_git_fallback_fails(
        self, env, monkeypatch, capsys, env_vars, git_results, message
    ):
        for key, value in env_vars.items():
            monkeypatch.setenv(key, value)
        env.urlopen(*[pr_fail(OSError("boom"))] * 5)
        env.git.results = list(git_results)

        with pytest.raises(SystemExit) as exc:
            env.run()
        out = capsys.readouterr().out

        assert exc.value.code == 1
        assert env.sleeps == [1, 2, 4, 8]
        assert env.git.results == []  # every queued git result was consumed
        assert "All 5 API attempts failed; trying the local git checkout" in out
        assert message in out
        assert "Could not get the PR diff from git either, exiting" in out
        assert "PR diff saved to" not in out
        assert not env.diff_file.exists()
        assert not env.output.exists()
