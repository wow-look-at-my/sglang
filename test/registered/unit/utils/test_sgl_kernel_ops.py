"""Every sgl_kernel op python/sglang/srt calls is defined in the AOT kernel source.

The check is static: it parses the TORCH_LIBRARY schema under
python/sglang/kernels/aot/csrc and the sgl_kernel Python package next to it, so an
op referenced from srt cannot land without its schema, and the runtime manifest in
sglang.srt.utils.sgl_kernel_ops cannot drift from either. XPU paths are excluded:
their sgl_kernel is the separately versioned sgl-kernel-xpu package.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

import ast
import re
import sys
import types
import unittest
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

import sglang
from sglang.srt.utils import sgl_kernel_ops
from sglang.test.test_utils import CustomTestCase

_PACKAGE_ROOT = Path(next(iter(sglang.__path__)))
_SRT_ROOT = _PACKAGE_ROOT / "srt"
_AOT_ROOT = _PACKAGE_ROOT / "kernels" / "aot"
_CSRC_ROOT = _AOT_ROOT / "csrc"
_PY_ROOT = _AOT_ROOT / "python" / "sgl_kernel"

# The library `import sgl_kernel` loads on each backend. flash_extension.cc,
# flashmla_extension.cc and spatial_extension.cc build separate libraries loaded
# only by their own submodules, so a startup probe cannot see their ops.
_MAIN_LIBRARY_SCHEMA = {
    "common_extension.cc": "cuda",
    "common_extension_rocm.cc": "rocm",
    "common_extension_musa.cc": "musa",
    "torch_extension_cpu.cpp": "cpu",
}

_LIBRARY_BLOCK = re.compile(
    r"TORCH_LIBRARY(?:_FRAGMENT|_EXPAND)?\(\s*sgl_kernel\s*,\s*(\w+)\s*\)"
)
_LINE_OR_BLOCK_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)


def _strip_cpp_comments(text: str) -> str:
    # Comments never hold a string literal containing // or /* in these files.
    return _LINE_OR_BLOCK_COMMENT.sub("", text)


@lru_cache(maxsize=1)
def _schema_ops() -> dict[str, frozenset[str]]:
    """op name -> the csrc files whose sgl_kernel library defines it."""
    defined = defaultdict(set)
    for path in sorted(_CSRC_ROOT.rglob("*")):
        if path.suffix not in (".cc", ".cpp", ".cu"):
            continue
        text = _strip_cpp_comments(path.read_text(errors="ignore"))
        for var in set(_LIBRARY_BLOCK.findall(text)):
            pattern = re.compile(rf"\b{re.escape(var)}\.def\(\s*\"(\w+)")
            for name in pattern.findall(text):
                defined[name].add(path.name)
    return {name: frozenset(files) for name, files in defined.items()}


def _is_sgl_kernel_module(name: str | None) -> bool:
    return name is not None and (name == "sgl_kernel" or name.startswith("sgl_kernel."))


def _module_path(module: str) -> Path | None:
    parts = module.split(".")[1:]
    base = _PY_ROOT.joinpath(*parts)
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    return None


@lru_cache(maxsize=None)
def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _module_level_statements(body):
    """Statements bound at module scope, including inside if / try / with."""
    for stmt in body:
        yield stmt
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for field in ("body", "orelse", "finalbody", "handlers"):
            children = getattr(stmt, field, None) or []
            yield from _module_level_statements(children)


def _torch_ops_name(node: ast.AST) -> str | None:
    """``X`` for ``torch.ops.sgl_kernel.X``."""
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "sgl_kernel"
        and isinstance(node.value.value, ast.Attribute)
        and node.value.value.attr == "ops"
        and isinstance(node.value.value.value, ast.Name)
        and node.value.value.value.id == "torch"
    ):
        return node.attr
    return None


class _Unresolved(Exception):
    pass


_RESOLVING: set[tuple[str, str]] = set()


@lru_cache(maxsize=None)
def _resolve(module: str, name: str) -> frozenset[str]:
    """Ops reached by ``from <module> import <name>``; raises _Unresolved if unbound."""
    if (module, name) in _RESOLVING:
        # Mutually recursive helpers: the caller already collects this one.
        return frozenset()
    _RESOLVING.add((module, name))
    try:
        return _resolve_binding(module, name)
    finally:
        _RESOLVING.discard((module, name))


def _resolve_binding(module: str, name: str) -> frozenset[str]:
    path = _module_path(module)
    if path is None:
        raise _Unresolved(f"{module} (no such module in the sgl_kernel source)")
    if _module_path(f"{module}.{name}") is not None:
        # A submodule reaches no op by itself; srt's attribute uses of it are
        # resolved one by one.
        return frozenset()
    tree = _parse(path)
    star_sources = []
    for stmt in _module_level_statements(tree.body):
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if stmt.name == name:
                return _ops_reached(module, stmt)
        elif isinstance(stmt, ast.ImportFrom):
            source = _absolute_module(module, stmt)
            for alias in stmt.names:
                if alias.name == "*":
                    if _is_sgl_kernel_module(source):
                        star_sources.append(source)
                elif (alias.asname or alias.name) == name:
                    if _is_sgl_kernel_module(source):
                        return _resolve(source, alias.name)
                    return frozenset()
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            for target in targets:
                if any(
                    isinstance(n, ast.Name) and n.id == name for n in ast.walk(target)
                ):
                    return _ops_reached(module, stmt)
    for source in star_sources:
        try:
            return _resolve(source, name)
        except _Unresolved:
            continue
    raise _Unresolved(f"{module}.{name}")


def _absolute_module(module: str, stmt: ast.ImportFrom) -> str | None:
    if not stmt.level:
        return stmt.module
    parts = module.split(".")
    if _module_path(module).name != "__init__.py":
        parts = parts[:-1]
    parts = parts[: len(parts) - (stmt.level - 1)]
    return ".".join(parts + ([stmt.module] if stmt.module else []))


def _ops_reached(module: str, node: ast.AST) -> frozenset[str]:
    """torch.ops.sgl_kernel names used by ``node``, following same-package helpers."""
    ops = set()
    local_names = set()
    for inner in ast.walk(node):
        op = _torch_ops_name(inner)
        if op is not None:
            ops.add(op)
        elif isinstance(inner, ast.Name) and isinstance(inner.ctx, ast.Load):
            local_names.add(inner.id)
    for helper in sorted(local_names):
        try:
            ops |= _resolve(module, helper)
        except _Unresolved:
            continue
    return frozenset(ops)


class _SrtReferenceCollector(ast.NodeVisitor):
    """sgl_kernel references in one srt module, skipping XPU-only scopes."""

    def __init__(self, rel: str):
        self.rel = rel
        self.xpu = "xpu" in rel.lower()
        self.torch_ops = []  # (op, location)
        self.imports = []  # (module, name, location)
        self.module_aliases = {}  # local name -> sgl_kernel module
        self.alias_attrs = []  # (alias, attr, location)

    def _scoped(self, xpu: bool, nodes):
        saved, self.xpu = self.xpu, self.xpu or xpu
        for node in nodes:
            self.visit(node)
        self.xpu = saved

    def visit_FunctionDef(self, node):
        self._scoped("xpu" in node.name.lower(), node.decorator_list + node.body)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self._scoped("xpu" in node.name.lower(), node.body)

    def visit_If(self, node):
        test = ast.unparse(node.test)
        self.visit(node.test)
        self._scoped("xpu" in test.lower() and not test.startswith("not "), node.body)
        self._scoped(False, node.orelse)

    def visit_ImportFrom(self, node):
        if self.xpu or not _is_sgl_kernel_module(node.module):
            return
        for alias in node.names:
            self.imports.append((node.module, alias.name, f"{self.rel}:{node.lineno}"))

    def visit_Import(self, node):
        if self.xpu:
            return
        for alias in node.names:
            if not _is_sgl_kernel_module(alias.name):
                continue
            if alias.asname:
                self.module_aliases[alias.asname] = alias.name
            else:
                self.module_aliases["sgl_kernel"] = "sgl_kernel"
            if _module_path(alias.name) is None:
                self.imports.append((alias.name, None, f"{self.rel}:{node.lineno}"))

    def visit_Attribute(self, node):
        if not self.xpu:
            op = _torch_ops_name(node)
            if op is not None:
                self.torch_ops.append((op, f"{self.rel}:{node.lineno}"))
            elif isinstance(node.value, ast.Name):
                self.alias_attrs.append(
                    (node.value.id, node.attr, f"{self.rel}:{node.lineno}")
                )
        self.generic_visit(node)


@lru_cache(maxsize=1)
def _srt_references():
    """(op -> locations, unresolved Python symbols, op -> Python symbols reaching it)."""
    direct_ops = defaultdict(list)
    unresolved = []
    via_wrappers = defaultdict(set)
    for path in sorted(_SRT_ROOT.rglob("*.py")):
        rel = path.relative_to(_PACKAGE_ROOT).as_posix()
        collector = _SrtReferenceCollector(rel)
        collector.visit(_parse(path))
        for op, where in collector.torch_ops:
            direct_ops[op].append(where)
        symbols = [(m, n, where) for m, n, where in collector.imports]
        for alias, attr, where in collector.alias_attrs:
            module = collector.module_aliases.get(alias)
            # Dunders (__version__, __file__) are module metadata, not bindings.
            if module is not None and not attr.startswith("__"):
                symbols.append((module, attr, where))
        for module, name, where in symbols:
            if name is None:
                unresolved.append(f"{where}: import {module}")
                continue
            try:
                for op in _resolve(module, name):
                    via_wrappers[op].add(f"{module}.{name}")
            except _Unresolved as exc:
                unresolved.append(f"{where}: {module}.{name} ({exc})")
    return direct_ops, unresolved, via_wrappers


def _expected_manifest() -> dict[str, frozenset[str]]:
    direct_ops, _, via_wrappers = _srt_references()
    reached = set(direct_ops) | set(via_wrappers)
    manifest = {backend: set() for backend in _MAIN_LIBRARY_SCHEMA.values()}
    for op, files in _schema_ops().items():
        if op not in reached:
            continue
        for file in files:
            backend = _MAIN_LIBRARY_SCHEMA.get(file)
            if backend is not None:
                manifest[backend].add(op)
    return {backend: frozenset(ops) for backend, ops in manifest.items()}


def _manifest_literal(manifest) -> str:
    lines = [
        "EXPECTED_SGL_KERNEL_OPS: Mapping[str, frozenset[str]] = MappingProxyType("
    ]
    lines.append("    {")
    for backend in sorted(manifest):
        ops = sorted(manifest[backend])
        if not ops:
            lines.append(f'        "{backend}": frozenset(),')
            continue
        lines.append(f'        "{backend}": frozenset(')
        lines.append("            {")
        lines.extend(f'                "{op}",' for op in ops)
        lines.append("            }")
        lines.append("        ),")
    lines.append("    }")
    lines.append(")")
    return "\n".join(lines)


class TestSglKernelOpsAgainstAotSource(CustomTestCase):
    def test_source_tree_is_present(self):
        # Guards the scan itself: an empty schema would pass every check below.
        self.assertTrue(_CSRC_ROOT.is_dir(), f"{_CSRC_ROOT} missing")
        self.assertIn("get_device_accessible_ptr", _schema_ops())

    def test_srt_torch_ops_are_defined_in_aot_schema(self):
        direct_ops, _, via_wrappers = _srt_references()
        schema = _schema_ops()
        missing = [
            f"torch.ops.sgl_kernel.{op} at {', '.join(where)}"
            for op, where in sorted(direct_ops.items())
            if op not in schema
        ] + [
            f"torch.ops.sgl_kernel.{op} via {', '.join(sorted(symbols))}"
            for op, symbols in sorted(via_wrappers.items())
            if op not in schema
        ]
        self.assertEqual(
            missing,
            [],
            "sgl_kernel ops referenced from python/sglang/srt have no TORCH_LIBRARY "
            "schema in python/sglang/kernels/aot/csrc; add the op there (and bump "
            "the kernel version) before calling it.",
        )

    def test_srt_sgl_kernel_imports_resolve(self):
        _, unresolved, _ = _srt_references()
        self.assertEqual(
            unresolved,
            [],
            "python/sglang/srt imports sgl_kernel symbols the AOT Python package "
            "(python/sglang/kernels/aot/python/sgl_kernel) does not define.",
        )

    def test_runtime_manifest_matches_source(self):
        expected = _expected_manifest()
        actual = {
            k: frozenset(v) for k, v in sgl_kernel_ops.EXPECTED_SGL_KERNEL_OPS.items()
        }
        self.assertEqual(
            actual,
            expected,
            "EXPECTED_SGL_KERNEL_OPS in python/sglang/srt/utils/sgl_kernel_ops.py "
            "is stale; replace it with:\n\n" + _manifest_literal(expected),
        )
        self.assertIn("get_device_accessible_ptr", expected["cuda"])


class TestWarnMissingSglKernelOps(CustomTestCase):
    def test_one_warning_names_every_absent_op(self):
        # Op names no build registers, so the probe misses them on any host.
        absent = ("zz_absent_op_b", "zz_absent_op_a")
        fake_wheel = types.SimpleNamespace(
            __version__="0.4.7", __file__="/x/sgl_kernel"
        )
        with (
            patch.object(sgl_kernel_ops, "_current_backend", return_value="cuda"),
            patch.object(
                sgl_kernel_ops,
                "EXPECTED_SGL_KERNEL_OPS",
                {"cuda": frozenset(absent)},
            ),
            patch.dict(sys.modules, {"sgl_kernel": fake_wheel}),
            self.assertLogs(sgl_kernel_ops.logger, level="WARNING") as logs,
        ):
            sgl_kernel_ops.warn_missing_sgl_kernel_ops()
        self.assertEqual(len(logs.records), 1)
        self.assertIn("0.4.7", logs.output[0])
        self.assertIn("2 op(s)", logs.output[0])
        self.assertIn("zz_absent_op_a, zz_absent_op_b", logs.output[0])

    def test_backend_without_aot_schema_is_silent(self):
        with patch.object(sgl_kernel_ops, "_current_backend", return_value=None):
            with self.assertNoLogs(sgl_kernel_ops.logger, level="WARNING"):
                sgl_kernel_ops.warn_missing_sgl_kernel_ops()


if __name__ == "__main__":
    unittest.main()
