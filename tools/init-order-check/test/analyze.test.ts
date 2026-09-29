import assert from "node:assert/strict";
import { readdirSync } from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { analyzeClass, classNames, createWorkspace, type Report } from "../src/analyze.ts";

const fixtures = path.join(path.dirname(fileURLToPath(import.meta.url)), "fixtures");
const workspace = await createWorkspace(fixtures);

interface Expect {
	file: string;
	// Attributes reported as read before assignment.
	reads: string[];
	// Attributes reported as never initialized by the constructor.
	uninitialized: string[];
	opaque?: boolean;
	dynamic?: boolean;
}

const CASES: Record<string, Expect> = {
	ReadBeforeAssign: { file: "pkg/order.py", reads: ["mode"], uninitialized: [] },
	Ordered: { file: "pkg/order.py", reads: [], uninitialized: [] },
	LazyField: { file: "pkg/order.py", reads: [], uninitialized: ["count"] },
	AsyncStep: { file: "pkg/order.py", reads: [], uninitialized: [] },
	GeneratorStep: { file: "pkg/order.py", reads: [], uninitialized: [] },
	AnnotationOnly: { file: "pkg/order.py", reads: [], uninitialized: [] },
	HasattrGuard: { file: "pkg/order.py", reads: [], uninitialized: [] },
	GetterReads: { file: "pkg/order.py", reads: ["base_value"], uninitialized: [] },
	SetterWrites: { file: "pkg/order.py", reads: [], uninitialized: [] },
	Mutual: { file: "pkg/order.py", reads: [], uninitialized: [] },
	ComputedSetattr: { file: "pkg/order.py", reads: [], uninitialized: [], dynamic: true },
	Derived: { file: "pkg/order.py", reads: ["lock"], uninitialized: [] },
	ExternalBase: { file: "pkg/order.py", reads: [], uninitialized: [], opaque: true },
	CooperativeMixin: { file: "pkg/order.py", reads: [], uninitialized: [], dynamic: true },
	Buffered: { file: "pkg/order.py", reads: [], uninitialized: [] },
	Middle: { file: "pkg/order.py", reads: ["later"], uninitialized: ["later"] },
	SkipsMiddle: { file: "pkg/order.py", reads: [], uninitialized: [] },
	NestedSelf: { file: "pkg/order.py", reads: [], uninitialized: [] },
	DevBase: { file: "pkg/order.py", reads: [], uninitialized: [] },
	ReadsDevice: { file: "pkg/order.py", reads: ["device"], uninitialized: ["device"] },
	SetsDevice: { file: "pkg/order.py", reads: [], uninitialized: [] },
	Diamond: { file: "pkg/order.py", reads: [], uninitialized: [] },
	HelperFills: { file: "pkg/order.py", reads: [], uninitialized: [] },
	SwapsClass: { file: "pkg/order.py", reads: [], uninitialized: [] },
	NoSplatMixin: { file: "pkg/order.py", reads: [], uninitialized: [], dynamic: true },
	CommentedParams: { file: "pkg/order.py", reads: ["size"], uninitialized: ["size"] },
	Resettable: { file: "pkg/order.py", reads: [], uninitialized: ["free"] },
	OverridesClear: { file: "pkg/order.py", reads: [], uninitialized: [] },
	ExplicitOther: { file: "pkg/order.py", reads: [], uninitialized: [] },
	AlwaysRaises: { file: "pkg/order.py", reads: [], uninitialized: [], dynamic: true },
	NamedSetter: { file: "pkg/order.py", reads: [], uninitialized: ["_paused"] },
	RaisesOnlyUnderCondition: { file: "pkg/order.py", reads: ["late"], uninitialized: ["late"] },
	Record: { file: "pkg/order.py", reads: [], uninitialized: ["free", "ready"] },
	UsesCycle: { file: "pkg/order.py", reads: [], uninitialized: [] },
	OnCycle: { file: "pkg/cycle_a.py", reads: ["second"], uninitialized: [] },
	Base: { file: "pkg/base.py", reads: [], uninitialized: [] },
};

function run(name: string, file: string): Report {
	const report = analyzeClass(workspace, path.join(fixtures, file), name);
	assert.ok(report, `${name} has an __init__, so it must be analyzed`);
	return report;
}

for (const [name, expect] of Object.entries(CASES)) {
	test(name, () => {
		const report = run(name, expect.file);
		assert.deepEqual([...new Set(report.findings.map((f) => f.attr))].sort(), [...expect.reads].sort(), "attributes read before assignment");
		assert.deepEqual(report.uninitialized.map((u) => u.attr).sort(), [...expect.uninitialized].sort(), "attributes the constructor never initializes");
		assert.equal(report.opaque, expect.opaque ?? false, "opaque");
		assert.equal(report.dynamic, expect.dynamic ?? false, "dynamic");
	});
}

test("a read reports the call chain and the first assignment", () => {
	const [finding] = run("ReadBeforeAssign", "pkg/order.py").findings;
	assert.deepEqual(finding?.chain, ["ReadBeforeAssign.__init__", "ReadBeforeAssign.init_throttle"]);
	assert.equal(finding?.assignedAt?.method, "ReadBeforeAssign.init_mode");
});

test("the base constructor's call to an override is followed", () => {
	const [finding] = run("Derived", "pkg/order.py").findings;
	assert.deepEqual(finding?.chain, ["Derived.__init__", "Base.__init__", "Derived.reset"]);
});

test("every fixture class is named by a case", () => {
	const pythonFiles = (dir: string): string[] =>
		readdirSync(dir, { withFileTypes: true }).flatMap((e) =>
			e.isDirectory() ? pythonFiles(path.join(dir, e.name)) : e.name.endsWith(".py") ? [path.join(dir, e.name)] : [],
		);
	const withInit = pythonFiles(fixtures).flatMap((file) =>
		classNames(workspace, file).filter((name) => analyzeClass(workspace, file, name) !== undefined),
	);
	assert.deepEqual(withInit.filter((n) => !(n in CASES)).sort(), []);
});
