"""Fail when a built sglang-kernel CUDA wheel lacks an op this SGLang tree calls.

Needs no GPU: every TORCH_LIBRARY m.def string is a literal in the wheel's
common_ops library, either a schema ("name(Tensor ...") or, for m.def("name", &fn),
the bare NUL-terminated name, so each expected op is looked up as bytes.
The expected set is EXPECTED_SGL_KERNEL_OPS["cuda"] from
python/sglang/srt/utils/sgl_kernel_ops.py, which a unit test keeps equal to the
AOT source.

    python3 scripts/ci/utils/check_sgl_kernel_wheel_ops.py dist/*.whl
"""

import argparse
import importlib.util
import sys
import zipfile
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST = REPO_ROOT / "python/sglang/srt/utils/sgl_kernel_ops.py"


def _expected_cuda_ops() -> frozenset[str]:
    # Loaded by path: importing sglang.srt.utils would pull in torch and the rest.
    spec = importlib.util.spec_from_file_location("_sgl_kernel_ops", MANIFEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EXPECTED_SGL_KERNEL_OPS["cuda"]


def _missing_ops(wheel: Path, expected: frozenset[str]) -> dict[str, list[str]]:
    """common_ops library path in the wheel -> expected ops absent from it."""
    missing = {}
    with zipfile.ZipFile(wheel) as archive:
        libraries = [
            name
            for name in archive.namelist()
            if PurePosixPath(name).name.startswith("common_ops")
            and name.endswith(".so")
        ]
        if not libraries:
            raise SystemExit(f"{wheel.name}: no common_ops*.so inside")
        for library in libraries:
            data = archive.read(library)
            absent = sorted(
                op
                for op in expected
                if f"{op}(".encode() not in data and f"{op}\0".encode() not in data
            )
            if absent:
                missing[library] = absent
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", nargs="+", type=Path)
    args = parser.parse_args()

    expected = _expected_cuda_ops()
    failed = False
    for wheel in args.wheels:
        missing = _missing_ops(wheel, expected)
        for library, ops in missing.items():
            failed = True
            print(f"{wheel.name}:{library} lacks {len(ops)} op(s): {', '.join(ops)}")
        if not missing:
            print(f"{wheel.name}: all {len(expected)} expected ops present")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
