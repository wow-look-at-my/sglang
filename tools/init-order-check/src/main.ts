// Checks constructor order of every class under python/sglang; exits non-zero on any finding.
import { readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { analyzeClass, classNames, createWorkspace, type Report } from "./analyze.ts";

const args = process.argv.slice(2);
let repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
let json = false;
const only: string[] = [];
for (let i = 0; i < args.length; i++) {
	const arg = args[i]!;
	if (arg === "--repo") repoRoot = path.resolve(args[++i] ?? "");
	else if (arg === "--json") json = true;
	else only.push(arg);
}

const packageRoot = path.join(repoRoot, "python");
const rel = (file: string) => path.relative(repoRoot, file);

function pythonFiles(dir: string): string[] {
	const out: string[] = [];
	for (const entry of readdirSync(dir, { withFileTypes: true })) {
		const full = path.join(dir, entry.name);
		if (entry.isDirectory()) out.push(...pythonFiles(full));
		else if (entry.name.endsWith(".py")) out.push(full);
	}
	return out.sort();
}

const files = only.length > 0 ? only.map((f) => path.resolve(repoRoot, f)) : pythonFiles(path.join(packageRoot, "sglang"));
const workspace = await createWorkspace(packageRoot);

interface Problem {
	file: string;
	line: number;
	className: string;
	classFile: string;
	message: string;
}

const problems: Problem[] = [];
let checked = 0;
let opaque = 0;
const partial: string[] = [];
const add = (report: Report, file: string, line: number, message: string) =>
	problems.push({ file: rel(file), line, className: report.className, classFile: rel(report.classFile), message });

for (const file of files) {
	for (const className of classNames(workspace, file)) {
		const report = analyzeClass(workspace, file, className);
		if (!report) continue;
		if (report.opaque) {
			opaque++;
			continue;
		}
		checked++;
		if (report.dynamic) partial.push(`${rel(report.classFile)}:${report.className}`);
		for (const f of report.findings) {
			const where = f.assignedAt ? `first assigned at ${rel(f.assignedAt.file)}:${f.assignedAt.line} in ${f.assignedAt.method}` : "never assigned";
			add(report, f.file, f.line, `self.${f.attr} is read before ${report.className}.__init__ assigns it (${where}; via ${f.chain.join(" -> ")})`);
		}
		for (const u of report.uninitialized) {
			add(report, u.assignedAt.file, u.assignedAt.line, `self.${u.attr} is assigned in ${u.assignedAt.method} but ${report.className}.__init__ never initializes it`);
		}
	}
}

// Each class reports its own problems, so a line shared through a base appears once per class.
if (json) {
	console.log(JSON.stringify(problems, null, "\t"));
} else {
	for (const p of problems) console.log(`${p.file}:${p.line}: ${p.message}`);
	console.log(`checked ${checked} classes in ${files.length} files: ${problems.length} problem(s)`);
	console.log(`not checked: ${opaque} classes with a base outside the repo that may set attributes`);
	console.log(`checked only up to code that can set any attribute (computed setattr, super() past the MRO): ${partial.length} classes`);
	for (const name of partial) console.log(`    ${name}`);
}
process.exitCode = problems.length > 0 ? 1 : 0;
