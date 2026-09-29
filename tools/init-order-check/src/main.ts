// Checks the constructor order of the large runtime classes; exits non-zero on any finding.
import path from "node:path";
import { fileURLToPath } from "node:url";
import { analyze, type Report, type Target } from "./analyze.ts";

const TARGETS: Target[] = [
	{ file: "python/sglang/srt/managers/scheduler.py", className: "Scheduler" },
	{ file: "python/sglang/srt/managers/tokenizer_manager.py", className: "TokenizerManager" },
	{ file: "python/sglang/srt/model_executor/model_runner.py", className: "ModelRunner" },
];

const repoRoot = path.resolve(process.argv[2] ?? path.join(path.dirname(fileURLToPath(import.meta.url)), "../../.."));
const rel = (file: string) => path.relative(repoRoot, file);

function print(report: Report): number {
	for (const f of report.findings) {
		const where = f.assignedAt ? `first assigned at ${rel(f.assignedAt.file)}:${f.assignedAt.line} in ${f.assignedAt.method}` : "never assigned";
		console.log(`${rel(f.file)}:${f.line}:${f.column}: self.${f.attr} is read before ${report.className}.__init__ assigns it (${where})`);
		console.log(`    via ${f.chain.join(" -> ")}`);
	}
	for (const u of report.uninitialized) {
		console.log(`${rel(u.assignedAt.file)}:${u.assignedAt.line}: self.${u.attr} is assigned in ${u.assignedAt.method} but ${report.className}.__init__ never initializes it`);
	}
	const problems = report.findings.length + report.uninitialized.length;
	const bases = report.unresolvedBases.length ? `; bases outside the repo: ${report.unresolvedBases.join(", ")}` : "";
	console.log(`${report.className}: walked ${report.inlinedMethods.length} methods, ${problems} problem(s)${bases}`);
	return problems;
}

let problems = 0;
for (const target of TARGETS) problems += print(await analyze(repoRoot, path.join(repoRoot, "python"), target));
process.exitCode = problems > 0 ? 1 : 0;
