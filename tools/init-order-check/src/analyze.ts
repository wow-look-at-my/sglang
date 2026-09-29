// Simulates a Python __init__ in source order and reports self attributes read before any earlier step assigned them.
import { existsSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { Language, Parser, type Node } from "web-tree-sitter";

export interface Location {
	file: string;
	line: number;
	method: string;
}

export interface Finding {
	file: string;
	line: number;
	column: number;
	attr: string;
	// The call chain from __init__ to the method that holds the read.
	chain: string[];
	// Where the attribute is first assigned, if any method assigns it.
	assignedAt: Location | undefined;
}

export interface Uninitialized {
	attr: string;
	// The first assignment outside the constructor path.
	assignedAt: Location;
}

export interface Report {
	className: string;
	findings: Finding[];
	// Instance attributes that some method assigns and no __init__ path assigns.
	uninitialized: Uninitialized[];
	// The method bodies the simulation walked; a small number means the check saw little.
	inlinedMethods: string[];
	unresolvedBases: string[];
}

interface Method {
	name: string;
	cls: ClassInfo;
	node: Node;
	selfName: string | undefined;
}

interface ClassInfo {
	name: string;
	file: string;
	bases: ClassInfo[];
	unresolvedBases: string[];
	methods: Map<string, Method>;
	getters: Map<string, Method>;
	setters: Map<string, Method>;
	classAttrs: Set<string>;
}

type Candidate = { kind: "class"; node: Node } | { kind: "import"; module: Node; name: string };

interface ModuleInfo {
	file: string;
	names: Map<string, Candidate[]>;
}

let parserPromise: Promise<Parser> | undefined;

function getParser(): Promise<Parser> {
	parserPromise ??= (async () => {
		await Parser.init();
		const require = createRequire(import.meta.url);
		const language = await Language.load(require.resolve("tree-sitter-python/tree-sitter-python.wasm"));
		const parser = new Parser();
		parser.setLanguage(language);
		return parser;
	})();
	return parserPromise;
}

function errorNodes(node: Node, out: Node[]): void {
	if (node.type === "ERROR" || node.isMissing) {
		out.push(node);
		return;
	}
	if (!node.hasError) return;
	for (const child of node.children) if (child) errorNodes(child, out);
}

function named(node: Node): Node[] {
	return node.namedChildren.filter((c): c is Node => c !== null);
}

function field(node: Node, name: string): Node | null {
	return node.childForFieldName(name);
}

class Workspace {
	private modules = new Map<string, ModuleInfo>();
	private classes = new Map<Node, ClassInfo>();

	private parser: Parser;
	// The directory that holds top-level packages, e.g. <repo>/python.
	private packageRoot: string;

	constructor(parser: Parser, packageRoot: string) {
		this.parser = parser;
		this.packageRoot = packageRoot;
	}

	module(file: string): ModuleInfo {
		const cached = this.modules.get(file);
		if (cached) return cached;
		const tree = this.parser.parse(readFileSync(file, "utf8"));
		if (!tree) throw new Error(`${file}: tree-sitter returned no tree`);
		const errors: Node[] = [];
		errorNodes(tree.rootNode, errors);
		if (errors.length > 0) {
			const at = errors.map((e) => `${e.startPosition.row + 1}:${e.startPosition.column + 1}`).join(", ");
			throw new Error(`${file}: syntax error at ${at}`);
		}
		const info: ModuleInfo = { file, names: new Map() };
		this.collectModuleNames(tree.rootNode, info);
		this.modules.set(file, info);
		return info;
	}

	// Top-level names, including those bound inside module-level if/try blocks.
	private collectModuleNames(node: Node, info: ModuleInfo): void {
		const add = (name: string, candidate: Candidate) => {
			const list = info.names.get(name) ?? [];
			list.push(candidate);
			info.names.set(name, list);
		};
		for (const child of named(node)) {
			switch (child.type) {
				case "class_definition":
					add(field(child, "name")!.text, { kind: "class", node: child });
					break;
				case "decorated_definition": {
					const def = field(child, "definition");
					if (def?.type === "class_definition") add(field(def, "name")!.text, { kind: "class", node: def });
					break;
				}
				case "import_from_statement": {
					const module = field(child, "module_name")!;
					for (const imported of child.childrenForFieldName("name")) {
						if (!imported) continue;
						if (imported.type === "aliased_import") {
							add(field(imported, "alias")!.text, { kind: "import", module, name: field(imported, "name")!.text });
						} else {
							add(imported.text, { kind: "import", module, name: imported.text });
						}
					}
					break;
				}
				case "if_statement":
				case "try_statement":
				case "elif_clause":
				case "else_clause":
				case "except_clause":
				case "finally_clause":
				case "block":
					this.collectModuleNames(child, info);
					break;
			}
		}
	}

	private resolveModule(fromFile: string, moduleNode: Node): string | undefined {
		let dir: string;
		let dotted: string;
		if (moduleNode.type === "relative_import") {
			const prefix = named(moduleNode).find((c) => c.type === "import_prefix");
			const dots = prefix ? prefix.text.length : 0;
			dir = path.dirname(fromFile);
			for (let i = 1; i < dots; i++) dir = path.dirname(dir);
			dotted = named(moduleNode).find((c) => c.type === "dotted_name")?.text ?? "";
		} else {
			dir = this.packageRoot;
			dotted = moduleNode.text;
		}
		const base = dotted ? path.join(dir, ...dotted.split(".")) : dir;
		for (const file of [`${base}.py`, path.join(base, "__init__.py")]) if (existsSync(file)) return file;
		return undefined;
	}

	// All class definitions a module-level name can refer to; empty when the name leaves the repo.
	resolveClass(file: string, name: string, depth = 0): { classes: ClassInfo[]; external: boolean } {
		if (depth > 20) throw new Error(`${file}: import cycle while resolving ${name}`);
		const candidates = this.module(file).names.get(name) ?? [];
		const classes: ClassInfo[] = [];
		let external = candidates.length === 0;
		for (const candidate of candidates) {
			if (candidate.kind === "class") {
				classes.push(this.classInfo(file, candidate.node));
				continue;
			}
			const target = this.resolveModule(file, candidate.module);
			if (!target) {
				external = true;
				continue;
			}
			const resolved = this.resolveClass(target, candidate.name, depth + 1);
			classes.push(...resolved.classes);
			external ||= resolved.external;
		}
		return { classes, external };
	}

	classInfo(file: string, node: Node): ClassInfo {
		const cached = this.classes.get(node);
		if (cached) return cached;
		const info: ClassInfo = {
			name: field(node, "name")!.text,
			file,
			bases: [],
			unresolvedBases: [],
			methods: new Map(),
			getters: new Map(),
			setters: new Map(),
			classAttrs: new Set(),
		};
		this.classes.set(node, info);
		const superclasses = field(node, "superclasses");
		for (const base of superclasses ? named(superclasses) : []) {
			if (base.type === "keyword_argument") continue;
			if (base.type !== "identifier") {
				info.unresolvedBases.push(base.text);
				continue;
			}
			const { classes, external } = this.resolveClass(file, base.text);
			info.bases.push(...classes);
			if (classes.length === 0 && external) info.unresolvedBases.push(base.text);
		}
		this.collectClassBody(field(node, "body")!, info);
		return info;
	}

	private collectClassBody(body: Node, info: ClassInfo): void {
		for (const child of named(body)) {
			if (child.type === "function_definition") {
				this.addMethod(info, child, []);
			} else if (child.type === "decorated_definition") {
				const def = field(child, "definition")!;
				const decorators = named(child)
					.filter((c) => c.type === "decorator")
					.map((d) => named(d)[0]?.text ?? "");
				if (def.type === "function_definition") this.addMethod(info, def, decorators);
				else if (def.type === "class_definition") info.classAttrs.add(field(def, "name")!.text);
			} else if (child.type === "class_definition") {
				info.classAttrs.add(field(child, "name")!.text);
			} else if (child.type === "expression_statement") {
				for (const expr of named(child)) {
					let assignment: Node | null = expr;
					while (assignment?.type === "assignment") {
						const left = field(assignment, "left")!;
						const right = field(assignment, "right");
						if (left.type === "identifier" && right) info.classAttrs.add(left.text);
						assignment = right;
					}
				}
			} else if (child.type === "if_statement" || child.type === "block" || child.type === "else_clause") {
				this.collectClassBody(child, info);
			}
		}
	}

	private addMethod(info: ClassInfo, def: Node, decorators: string[]): void {
		const name = field(def, "name")!.text;
		const isStatic = decorators.includes("staticmethod");
		const params = named(field(def, "parameters")!);
		const first = params[0];
		let selfName: string | undefined;
		if (!isStatic && first) {
			if (first.type === "identifier") selfName = first.text;
			else if (first.type === "typed_parameter") selfName = named(first).find((c) => c.type === "identifier")?.text;
		}
		const method: Method = { name, cls: info, node: def, selfName };
		if (decorators.some((d) => d === "property" || d.endsWith("cached_property"))) {
			info.getters.set(name, method);
		} else if (decorators.some((d) => d === `${name}.setter`)) {
			info.setters.set(name, method);
		} else {
			// A later definition of the same name wins, as it does at runtime.
			info.methods.set(name, method);
		}
	}
}

// A left-to-right depth-first MRO; close enough to C3 for the mixin chains this runs on.
function linearize(cls: ClassInfo): ClassInfo[] {
	const out: ClassInfo[] = [];
	const visit = (c: ClassInfo) => {
		if (out.includes(c)) return;
		out.push(c);
		for (const base of c.bases) visit(base);
	};
	visit(cls);
	return out;
}

interface State {
	assigned: Set<string>;
	guarded: Set<string>;
}

interface Context {
	method: Method;
	selfName: string;
	chain: string[];
}

function cloneState(s: State): State {
	return { assigned: new Set(s.assigned), guarded: new Set(s.guarded) };
}

function stringLiteral(node: Node | undefined): string | undefined {
	if (node?.type !== "string") return undefined;
	const parts = named(node);
	if (parts.some((p) => p.type === "interpolation")) return undefined;
	return parts
		.filter((p) => p.type === "string_content")
		.map((p) => p.text)
		.join("");
}

function callArgs(call: Node): Node[] {
	const args = field(call, "arguments");
	return args ? named(args) : [];
}

class Simulator {
	private mro: ClassInfo[];
	private known = new Map<string, Location>();
	private findings = new Map<string, Finding>();
	private inlined = new Set<Method>();
	private stack: Method[] = [];
	private writesMemo = new Map<Method, Set<string>>();
	private cls: ClassInfo;

	constructor(cls: ClassInfo) {
		this.cls = cls;
		this.mro = linearize(cls);
		for (const c of this.mro) {
			for (const m of [...c.methods.values(), ...c.getters.values(), ...c.setters.values()]) this.collectKnown(m);
		}
	}

	run(): Report {
		const init = this.findMethod("__init__");
		if (!init) throw new Error(`${this.cls.file}: ${this.cls.name} has no __init__ in its resolved MRO`);
		const state: State = { assigned: new Set(), guarded: new Set() };
		this.inline(init, state, []);
		const uninitialized: Uninitialized[] = [];
		for (const [attr, assignedAt] of this.known) {
			if (!state.assigned.has(attr) && !this.isClassLevel(attr) && !this.findSetter(attr)) uninitialized.push({ attr, assignedAt });
		}
		return {
			className: this.cls.name,
			findings: [...this.findings.values()].sort((a, b) => a.file.localeCompare(b.file) || a.line - b.line),
			uninitialized: uninitialized.sort((a, b) => a.assignedAt.file.localeCompare(b.assignedAt.file) || a.assignedAt.line - b.assignedAt.line),
			inlinedMethods: [...this.inlined].map((m) => `${m.cls.name}.${m.name}`),
			unresolvedBases: this.mro.flatMap((c) => c.unresolvedBases.map((b) => `${c.name}(${b})`)),
		};
	}

	private findMethod(name: string, after?: ClassInfo): Method | undefined {
		let classes = this.mro;
		if (after) classes = classes.slice(classes.indexOf(after) + 1);
		for (const c of classes) {
			const m = c.methods.get(name);
			if (m) return m;
		}
		return undefined;
	}

	private findGetter(name: string): Method | undefined {
		for (const c of this.mro) {
			if (c.getters.has(name)) return c.getters.get(name);
			if (c.methods.has(name) || c.classAttrs.has(name)) return undefined;
		}
		return undefined;
	}

	private findSetter(name: string): Method | undefined {
		for (const c of this.mro) if (c.setters.has(name)) return c.setters.get(name);
		return undefined;
	}

	private isClassLevel(name: string): boolean {
		return this.mro.some((c) => c.methods.has(name) || c.getters.has(name) || c.classAttrs.has(name));
	}

	private label(m: Method): string {
		return `${m.cls.name}.${m.name}`;
	}

	// Every self.<attr> target in any method, nested functions included, so the check knows which names are instance attributes.
	private collectKnown(m: Method): void {
		const selfName = m.selfName;
		if (!selfName) return;
		const note = (attr: string, node: Node) => {
			if (!this.known.has(attr)) this.known.set(attr, { file: m.cls.file, line: node.startPosition.row + 1, method: this.label(m) });
		};
		const walk = (node: Node) => {
			if (node.type === "assignment" || node.type === "augmented_assignment") {
				this.targetAttrs(field(node, "left")!, selfName, note);
			} else if (node.type === "for_statement" || node.type === "for_in_clause") {
				this.targetAttrs(field(node, "left")!, selfName, note);
			} else if (node.type === "as_pattern") {
				const target = named(node).find((c) => c.type === "as_pattern_target");
				if (target) for (const t of named(target)) this.targetAttrs(t, selfName, note);
			} else if (node.type === "call") {
				const fn = field(node, "function");
				const args = callArgs(node);
				const attr = stringLiteral(args[1]);
				if (fn?.text === "setattr" && args[0]?.text === selfName && attr) note(attr, node);
			}
			for (const child of named(node)) walk(child);
		};
		walk(field(m.node, "body")!);
	}

	private targetAttrs(target: Node, selfName: string, cb: (attr: string, node: Node) => void): void {
		if (target.type === "attribute") {
			const obj = field(target, "object")!;
			if (obj.type === "identifier" && obj.text === selfName) cb(field(target, "attribute")!.text, target);
			return;
		}
		if (["pattern_list", "tuple_pattern", "list_pattern", "tuple", "list", "parenthesized_expression", "list_splat_pattern"].includes(target.type)) {
			for (const child of named(target)) this.targetAttrs(child, selfName, cb);
		}
	}

	// Attributes a method assigns on the straight-line path, used when the method was already walked once.
	private writesOf(m: Method): Set<string> {
		const memo = this.writesMemo.get(m);
		if (memo) return memo;
		const out = new Set<string>();
		this.writesMemo.set(m, out);
		const selfName = m.selfName;
		if (!selfName) return out;
		const walk = (node: Node) => {
			if (node.type === "function_definition" || node.type === "lambda" || node.type === "class_definition") return;
			if (node.type === "assignment" || node.type === "augmented_assignment" || node.type === "for_statement") {
				this.targetAttrs(field(node, "left")!, selfName, (attr) => {
					const setter = this.findSetter(attr);
					if (setter) for (const w of this.writesOf(setter)) out.add(w);
					else out.add(attr);
				});
			} else if (node.type === "call") {
				const callee = this.calleeOf(node, { method: m, selfName, chain: [] });
				if (callee) for (const w of this.writesOf(callee)) out.add(w);
				const args = callArgs(node);
				const attr = stringLiteral(args[1]);
				if (field(node, "function")?.text === "setattr" && args[0]?.text === selfName && attr) out.add(attr);
			}
			for (const child of named(node)) walk(child);
		};
		walk(field(m.node, "body")!);
		return out;
	}

	// The method a call runs on this instance: self.m(), super().m(), or Base.m(self).
	private calleeOf(call: Node, ctx: Context): Method | undefined {
		const fn = field(call, "function");
		if (fn?.type !== "attribute") return undefined;
		const obj = field(fn, "object")!;
		const name = field(fn, "attribute")!.text;
		if (obj.type === "identifier" && obj.text === ctx.selfName) return this.findMethod(name);
		if (obj.type === "call" && field(obj, "function")?.text === "super") return this.findMethod(name, ctx.method.cls);
		if (obj.type === "identifier" && callArgs(call)[0]?.text === ctx.selfName) {
			const cls = this.mro.find((c) => c.name === obj.text);
			if (cls) {
				for (const c of this.mro.slice(this.mro.indexOf(cls))) {
					const m = c.methods.get(name);
					if (m) return m;
				}
			}
		}
		return undefined;
	}

	private inline(m: Method, state: State, chain: string[]): void {
		if (this.stack.includes(m) || !m.selfName) return;
		if (this.inlined.has(m)) {
			for (const w of this.writesOf(m)) state.assigned.add(w);
			return;
		}
		this.inlined.add(m);
		this.stack.push(m);
		const ctx: Context = { method: m, selfName: m.selfName, chain: [...chain, this.label(m)] };
		this.visitBlock(field(m.node, "body")!, state, ctx);
		this.stack.pop();
	}

	private visitBlock(block: Node, state: State, ctx: Context): void {
		for (const stmt of named(block)) this.visitStatement(stmt, state, ctx);
	}

	private visitStatement(node: Node, state: State, ctx: Context): void {
		switch (node.type) {
			case "function_definition":
			case "class_definition":
			case "decorated_definition":
			case "comment":
			case "import_statement":
			case "import_from_statement":
			case "future_import_statement":
			case "global_statement":
			case "nonlocal_statement":
			case "pass_statement":
			case "break_statement":
			case "continue_statement":
				return;
			case "block":
				this.visitBlock(node, state, ctx);
				return;
			case "if_statement":
				this.visitIf(node, state, ctx);
				return;
			case "for_statement": {
				this.visitExpr(field(node, "right")!, state, ctx);
				this.recordTarget(field(node, "left")!, state, ctx);
				this.visitBlock(field(node, "body")!, state, ctx);
				const alt = field(node, "alternative");
				if (alt) this.visitBlock(field(alt, "body")!, state, ctx);
				return;
			}
			case "while_statement": {
				this.visitExpr(field(node, "condition")!, state, ctx);
				this.visitBlock(field(node, "body")!, state, ctx);
				const alt = field(node, "alternative");
				if (alt) this.visitBlock(field(alt, "body")!, state, ctx);
				return;
			}
			case "try_statement":
				// Handlers and finally run after some prefix of the body; treating the body as complete is optimistic.
				for (const child of named(node)) {
					if (child.type === "block") this.visitBlock(child, state, ctx);
					else if (child.type === "except_clause" || child.type === "except_group_clause") {
						const body = named(child).find((c) => c.type === "block");
						if (body) this.visitBlock(body, state, ctx);
					} else if (child.type === "else_clause" || child.type === "finally_clause") {
						const body = field(child, "body") ?? named(child).find((c) => c.type === "block");
						if (body) this.visitBlock(body, state, ctx);
					}
				}
				return;
			case "with_statement":
				for (const child of named(node)) {
					if (child.type === "with_clause") for (const item of named(child)) this.visitExpr(item, state, ctx);
					else if (child.type === "block") this.visitBlock(child, state, ctx);
				}
				return;
			case "match_statement": {
				this.visitExpr(field(node, "subject")!, state, ctx);
				const results: State[] = [];
				for (const clause of named(field(node, "body")!)) {
					const branch = cloneState(state);
					const body = field(clause, "consequence") ?? named(clause).find((c) => c.type === "block");
					if (body) this.visitBlock(body, branch, ctx);
					results.push(branch);
				}
				for (const r of results) for (const a of r.assigned) state.assigned.add(a);
				return;
			}
			default:
				for (const child of named(node)) this.visitExpr(child, state, ctx);
		}
	}

	private visitIf(node: Node, state: State, ctx: Context): void {
		const condition = field(node, "condition")!;
		const guards = this.hasattrGuards(condition, ctx);
		const guarded = cloneState(state);
		for (const g of guards) guarded.guarded.add(g);
		this.visitExpr(condition, guarded, ctx);
		for (const a of guarded.assigned) state.assigned.add(a);

		const branches: State[] = [];
		const run = (body: Node) => {
			const branch = cloneState(state);
			for (const g of guards) branch.guarded.add(g);
			this.visitBlock(body, branch, ctx);
			branch.guarded = new Set(state.guarded);
			branches.push(branch);
		};
		run(field(node, "consequence")!);
		let hasElse = false;
		for (const alt of node.childrenForFieldName("alternative")) {
			if (!alt) continue;
			if (alt.type === "elif_clause") {
				const elifCondition = field(alt, "condition")!;
				const elifGuards = this.hasattrGuards(elifCondition, ctx);
				const branchState = cloneState(state);
				for (const g of [...guards, ...elifGuards]) branchState.guarded.add(g);
				this.visitExpr(elifCondition, branchState, ctx);
				this.visitBlock(field(alt, "consequence")!, branchState, ctx);
				branchState.guarded = new Set(state.guarded);
				branches.push(branchState);
			} else {
				hasElse = true;
				run(field(alt, "body")!);
			}
		}
		if (!hasElse) branches.push(cloneState(state));
		for (const b of branches) for (const a of b.assigned) state.assigned.add(a);
	}

	// Names checked with hasattr(self, "x") in a condition; reads under that condition are not reported.
	private hasattrGuards(node: Node, ctx: Context): string[] {
		const out: string[] = [];
		const walk = (n: Node) => {
			if (n.type === "call" && field(n, "function")?.text === "hasattr") {
				const args = callArgs(n);
				const attr = stringLiteral(args[1]);
				if (args[0]?.text === ctx.selfName && attr) out.push(attr);
			}
			for (const child of named(n)) walk(child);
		};
		walk(node);
		return out;
	}

	private visitExpr(node: Node | null, state: State, ctx: Context): void {
		if (!node) return;
		switch (node.type) {
			case "lambda":
				return;
			case "assignment": {
				const right = field(node, "right");
				if (right) this.visitExpr(right, state, ctx);
				this.recordTarget(field(node, "left")!, state, ctx);
				return;
			}
			case "augmented_assignment": {
				const left = field(node, "left")!;
				this.visitExpr(left, state, ctx);
				this.visitExpr(field(node, "right")!, state, ctx);
				this.recordTarget(left, state, ctx);
				return;
			}
			case "attribute": {
				const obj = field(node, "object")!;
				if (obj.type === "identifier" && obj.text === ctx.selfName) this.read(field(node, "attribute")!.text, node, state, ctx);
				else this.visitExpr(obj, state, ctx);
				return;
			}
			case "call":
				this.visitCall(node, state, ctx);
				return;
			case "boolean_operator": {
				const left = field(node, "left")!;
				this.visitExpr(left, state, ctx);
				const branch = cloneState(state);
				if (field(node, "operator")?.text === "and") for (const g of this.hasattrGuards(left, ctx)) branch.guarded.add(g);
				this.visitExpr(field(node, "right")!, branch, ctx);
				for (const a of branch.assigned) state.assigned.add(a);
				return;
			}
			case "conditional_expression": {
				const [whenTrue, condition, whenFalse] = named(node);
				this.visitExpr(condition ?? null, state, ctx);
				const guards = condition ? this.hasattrGuards(condition, ctx) : [];
				const branch = cloneState(state);
				for (const g of guards) branch.guarded.add(g);
				this.visitExpr(whenTrue ?? null, branch, ctx);
				this.visitExpr(whenFalse ?? null, state, ctx);
				for (const a of branch.assigned) state.assigned.add(a);
				return;
			}
			case "keyword_argument":
				this.visitExpr(field(node, "value"), state, ctx);
				return;
			case "as_pattern": {
				const [value] = named(node);
				this.visitExpr(value ?? null, state, ctx);
				const target = named(node).find((c) => c.type === "as_pattern_target");
				if (target) for (const t of named(target)) this.recordTarget(t, state, ctx);
				return;
			}
			default:
				for (const child of named(node)) this.visitExpr(child, state, ctx);
		}
	}

	private visitCall(node: Node, state: State, ctx: Context): void {
		const fn = field(node, "function")!;
		const args = callArgs(node);
		if (fn.type === "identifier" && ["getattr", "hasattr", "setattr"].includes(fn.text) && args[0]?.text === ctx.selfName) {
			const attr = stringLiteral(args[1]);
			if (fn.text === "getattr" && args.length === 2 && attr) this.read(attr, node, state, ctx);
			for (const arg of args.slice(1)) this.visitExpr(arg, state, ctx);
			if (fn.text === "setattr" && attr) state.assigned.add(attr);
			return;
		}
		const callee = this.calleeOf(node, ctx);
		if (callee) {
			for (const arg of args) this.visitExpr(arg, state, ctx);
			this.inline(callee, state, ctx.chain);
			return;
		}
		this.visitExpr(fn, state, ctx);
		for (const arg of args) this.visitExpr(arg, state, ctx);
	}

	private recordTarget(target: Node, state: State, ctx: Context): void {
		switch (target.type) {
			case "identifier":
				return;
			case "attribute": {
				const obj = field(target, "object")!;
				if (obj.type === "identifier" && obj.text === ctx.selfName) {
					const attr = field(target, "attribute")!.text;
					const setter = this.findSetter(attr);
					if (setter) this.inline(setter, state, ctx.chain);
					else state.assigned.add(attr);
				} else {
					this.visitExpr(obj, state, ctx);
				}
				return;
			}
			case "subscript":
				for (const child of named(target)) this.visitExpr(child, state, ctx);
				return;
			default:
				for (const child of named(target)) this.recordTarget(child, state, ctx);
		}
	}

	private read(attr: string, node: Node, state: State, ctx: Context): void {
		if (state.assigned.has(attr) || state.guarded.has(attr)) return;
		const getter = this.findGetter(attr);
		if (getter) {
			this.inline(getter, state, ctx.chain);
			return;
		}
		if (this.isClassLevel(attr) || !this.known.has(attr)) return;
		const file = ctx.method.cls.file;
		const key = `${file}:${node.startPosition.row}:${attr}`;
		if (this.findings.has(key)) return;
		this.findings.set(key, {
			file,
			line: node.startPosition.row + 1,
			column: node.startPosition.column + 1,
			attr,
			chain: ctx.chain,
			assignedAt: this.known.get(attr),
		});
	}
}

export interface Target {
	// Relative to the repo root.
	file: string;
	className: string;
}

export async function analyze(repoRoot: string, packageRoot: string, target: Target): Promise<Report> {
	const workspace = new Workspace(await getParser(), packageRoot);
	const file = path.join(repoRoot, target.file);
	const { classes } = workspace.resolveClass(file, target.className);
	const cls = classes.find((c) => c.file === file);
	if (!cls) throw new Error(`${target.file}: no class ${target.className}`);
	return new Simulator(cls).run();
}
