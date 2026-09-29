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
	classFile: string;
	findings: Finding[];
	// Instance attributes that some method assigns and no __init__ path assigns.
	uninitialized: Uninitialized[];
	// The method bodies the simulation walked; a small number means the check saw little.
	inlinedMethods: string[];
	unresolvedBases: string[];
	// True when a base outside the repo may set attributes, so no read or omission is reported.
	opaque: boolean;
	// True when the constructor path can assign any name; reads after that point are not reported.
	dynamic: boolean;
	partialReason: string | undefined;
}

interface Method {
	name: string;
	// For a module function that receives the instance, the class whose constructor called it.
	cls: ClassInfo;
	file: string;
	label: string;
	node: Node;
	selfName: string | undefined;
	// An async or generator body does not run when called; the call only builds a coroutine or generator.
	deferred: boolean;
}

function paramName(param: Node | undefined): string | undefined {
	if (!param) return undefined;
	if (param.type === "identifier") return param.text;
	const name = field(param, "name");
	if (name) return name.text;
	return named(param).find((c) => c.type === "identifier")?.text;
}

function isDeferred(def: Node): boolean {
	if (def.children.some((c) => c?.type === "async")) return true;
	const hasYield = (node: Node): boolean => {
		if (node.type === "yield") return true;
		if (node.type === "function_definition" || node.type === "lambda" || node.type === "class_definition") return false;
		return named(node).some(hasYield);
	};
	return hasYield(field(def, "body")!);
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
	// A @dataclass without its own __init__ gets a generated one: fields, then __post_init__.
	dataclass: boolean;
}

type Candidate =
	| { kind: "class"; node: Node }
	| { kind: "function"; node: Node }
	| { kind: "import"; module: Node; name: string };

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

// Named children without comments, which tree-sitter places anywhere, even inside a parameter list.
function named(node: Node): Node[] {
	return node.namedChildren.filter((c): c is Node => c !== null && c.type !== "comment");
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

	definedClasses(file: string): string[] {
		const names: string[] = [];
		for (const [name, candidates] of this.module(file).names) {
			if (candidates.some((c) => c.kind === "class")) names.push(name);
		}
		return names;
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
				case "function_definition":
					add(field(child, "name")!.text, { kind: "function", node: child });
					break;
				case "decorated_definition": {
					const def = field(child, "definition");
					if (def?.type === "class_definition") add(field(def, "name")!.text, { kind: "class", node: def });
					if (def?.type === "function_definition") add(field(def, "name")!.text, { kind: "function", node: def });
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
	resolveClass(file: string, name: string, visiting = new Set<string>()): { classes: ClassInfo[]; external: boolean } {
		// A cycle of re-exports, often through TYPE_CHECKING imports, defines nothing on that path.
		const key = `${file}\0${name}`;
		if (visiting.has(key)) return { classes: [], external: false };
		visiting.add(key);
		const candidates = this.module(file).names.get(name) ?? [];
		const classes: ClassInfo[] = [];
		let external = candidates.length === 0;
		for (const candidate of candidates) {
			if (candidate.kind === "function") continue;
			if (candidate.kind === "class") {
				classes.push(this.classInfo(file, candidate.node));
				continue;
			}
			const target = this.resolveModule(file, candidate.module);
			if (!target) {
				external = true;
				continue;
			}
			const resolved = this.resolveClass(target, candidate.name, visiting);
			classes.push(...resolved.classes);
			external ||= resolved.external;
		}
		return { classes, external };
	}

	// The module-level function a name refers to, followed through imports.
	resolveFunction(file: string, name: string, visiting = new Set<string>()): { file: string; node: Node } | undefined {
		const key = `${file}\0${name}`;
		if (visiting.has(key)) return undefined;
		visiting.add(key);
		for (const candidate of this.module(file).names.get(name) ?? []) {
			if (candidate.kind === "function") return { file, node: candidate.node };
			if (candidate.kind !== "import") continue;
			const target = this.resolveModule(file, candidate.module);
			const found = target ? this.resolveFunction(target, candidate.name, visiting) : undefined;
			if (found) return found;
		}
		return undefined;
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
			dataclass: false,
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
		const decorated = node.parent?.type === "decorated_definition" ? node.parent : null;
		const dataclass = decorated !== null && named(decorated).some((d) => d.type === "decorator" && /\bdataclass\b/.test(d.text));
		info.dataclass = dataclass;
		this.collectClassBody(field(node, "body")!, info, dataclass);
		return info;
	}

	// In a @dataclass body, `x: T` declares a field that the generated __init__ sets.
	private collectClassBody(body: Node, info: ClassInfo, dataclass: boolean): void {
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
						if (left.type === "identifier" && (right || dataclass)) info.classAttrs.add(left.text);
						assignment = right;
					}
				}
			} else if (child.type === "if_statement" || child.type === "block" || child.type === "else_clause") {
				this.collectClassBody(child, info, dataclass);
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
		const method: Method = { name, cls: info, file: info.file, label: `${info.name}.${name}`, node: def, selfName, deferred: isDeferred(def) };
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

// Python's C3 MRO. An inconsistent hierarchy, which Python itself rejects, falls back to the first head.
function linearize(cls: ClassInfo, visiting = new Set<ClassInfo>()): ClassInfo[] {
	if (visiting.has(cls)) return [cls];
	visiting.add(cls);
	const seqs = [...cls.bases.map((b) => linearize(b, visiting)), [...cls.bases]].map((s) => [...s]);
	visiting.delete(cls);
	const out = [cls];
	for (;;) {
		const open = seqs.filter((s) => s.length > 0);
		if (open.length === 0) return out;
		const head = open.map((s) => s[0]!).find((h) => !open.some((s) => s.indexOf(h) > 0)) ?? open[0]![0]!;
		if (!out.includes(head)) out.push(head);
		for (const s of open) if (s[0] === head) s.shift();
	}
}

interface State {
	assigned: Set<string>;
	guarded: Set<string>;
}

interface Context {
	method: Method;
	selfName: string;
	chain: string[];
	// Parameters bound to a string literal at the call, e.g. name="_server_status".
	literals: Map<string, string>;
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

// Bases outside the repo that set no attributes a subclass reads.
const EMPTY_BASES = /^(object|ABC|abc\.ABC|nn\.Module|torch\.nn\.Module|(typing\.)?(Generic|Protocol)(\[.*\])?)$/s;

// An assignment with a value; `self.x: T` alone declares a type and binds nothing.
function binds(node: Node): boolean {
	return node.type === "augmented_assignment" || (node.type === "assignment" && field(node, "right") !== null);
}

// A nested class, or a nested function with its own `self` parameter, refers to a different object.
function shadows(node: Node, selfName: string): boolean {
	if (node.type === "class_definition") return true;
	if (node.type !== "function_definition" && node.type !== "lambda") return false;
	const params = field(node, "parameters");
	if (!params) return false;
	return named(params).some((p) => (p.type === "identifier" ? p : named(p).find((c) => c.type === "identifier"))?.text === selfName);
}

// nn.Module methods that bind an attribute under the name in their first argument.
const REGISTERS = new Set(["register_buffer", "register_parameter", "register_module", "add_module"]);

// The attribute a call binds by a literal name: setattr(self, "x", v) or self.register_buffer("x", t).
function namedWrite(call: Node, selfName: string): string | undefined {
	const fn = field(call, "function");
	const args = callArgs(call);
	if (fn?.text === "setattr" && args[0]?.text === selfName) return stringLiteral(args[1]);
	if (fn?.type !== "attribute" || field(fn, "object")?.text !== selfName) return undefined;
	return REGISTERS.has(field(fn, "attribute")!.text) ? stringLiteral(args[0]) : undefined;
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
	private workspace: Workspace;
	private helpers = new Map<string, Method>();
	// A base outside the repo, such as nn.Conv2d, can set any attribute; the check then trusts every read.
	private opaque: boolean;
	// Set once the constructor path can assign any name. Later reads and omissions are then trusted.
	private dynamic = false;
	private partialReason: string | undefined;
	// Nesting of conditional code on the current path; a raise inside it does not end the constructor.
	private branchDepth = 0;

	private markPartial(reason: string): void {
		this.dynamic = true;
		this.partialReason ??= reason;
	}

	constructor(cls: ClassInfo, workspace: Workspace) {
		this.cls = cls;
		this.workspace = workspace;
		this.mro = linearize(cls);
		this.opaque = this.mro.some((c) => c.unresolvedBases.some((b) => !EMPTY_BASES.test(b)));
		for (const c of this.mro) {
			for (const m of [...c.methods.values(), ...c.getters.values(), ...c.setters.values()]) {
				if (this.runs(m)) this.collectKnown(m);
			}
		}
	}

	// A base method that this class overrides runs only if the override's own super() chain reaches it.
	private runs(m: Method): boolean {
		if (!m.cls.methods.has(m.name)) return true;
		let current = this.findMethod(m.name);
		const seen = new Set<Method>();
		while (current && !seen.has(current)) {
			if (current === m) return true;
			seen.add(current);
			const calls = new RegExp(`super\\([^)]*\\)\\s*\\.\\s*${m.name}\\s*\\(`).test(current.node.text);
			current = calls ? this.findMethod(m.name, current.cls) : undefined;
		}
		return false;
	}

	private generatedInit(): boolean {
		return this.cls.dataclass && !this.cls.methods.has("__init__");
	}

	hasInit(): boolean {
		return this.generatedInit() || this.findMethod("__init__") !== undefined;
	}

	run(): Report {
		const state: State = { assigned: new Set(), guarded: new Set() };
		if (this.generatedInit()) {
			const post = this.findMethod("__post_init__");
			if (post) this.inline(post, state, []);
		} else {
			const init = this.findMethod("__init__");
			if (!init) throw new Error(`${this.cls.file}: ${this.cls.name} has no __init__ in its resolved MRO`);
			this.inline(init, state, []);
		}
		const uninitialized: Uninitialized[] = [];
		for (const [attr, assignedAt] of this.known) {
			if (this.opaque || this.dynamic || state.assigned.has(attr) || this.isClassLevel(attr) || this.findSetter(attr)) continue;
			uninitialized.push({ attr, assignedAt });
		}
		return {
			className: this.cls.name,
			classFile: this.cls.file,
			findings: [...this.findings.values()].sort((a, b) => a.file.localeCompare(b.file) || a.line - b.line),
			uninitialized: uninitialized.sort((a, b) => a.assignedAt.file.localeCompare(b.assignedAt.file) || a.assignedAt.line - b.assignedAt.line),
			inlinedMethods: [...this.inlined].map((m) => `${m.cls.name}.${m.name}`),
			unresolvedBases: this.mro.flatMap((c) => c.unresolvedBases.map((b) => `${c.name}(${b})`)),
			opaque: this.opaque,
			dynamic: this.dynamic,
			partialReason: this.partialReason,
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
		return m.label;
	}

	// Every self.<attr> target in any method, nested functions included, so the check knows which names are instance attributes.
	private collectKnown(m: Method): void {
		const selfName = m.selfName;
		if (!selfName) return;
		const note = (attr: string, node: Node) => {
			if (!this.known.has(attr)) this.known.set(attr, { file: m.cls.file, line: node.startPosition.row + 1, method: this.label(m) });
		};
		const walk = (node: Node) => {
			if (shadows(node, selfName)) return;
			if (binds(node)) {
				this.targetAttrs(field(node, "left")!, selfName, note);
			} else if (node.type === "for_statement" || node.type === "for_in_clause") {
				this.targetAttrs(field(node, "left")!, selfName, note);
			} else if (node.type === "as_pattern") {
				const target = named(node).find((c) => c.type === "as_pattern_target");
				if (target) for (const t of named(target)) this.targetAttrs(t, selfName, note);
			} else if (node.type === "call") {
				const attr = namedWrite(node, selfName);
				if (attr) note(attr, node);
			}
			for (const child of named(node)) walk(child);
		};
		walk(field(m.node, "body")!);
	}

	private targetAttrs(target: Node, selfName: string, cb: (attr: string, node: Node) => void): void {
		if (target.type === "attribute") {
			const obj = field(target, "object")!;
			const attr = field(target, "attribute")!.text;
			// self.__class__ = X swaps the class; it binds no field.
			if (obj.type === "identifier" && obj.text === selfName && !/^__\w+__$/.test(attr)) cb(attr, target);
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
			if (binds(node) || node.type === "for_statement") {
				this.targetAttrs(field(node, "left")!, selfName, (attr) => {
					const setter = this.findSetter(attr);
					if (setter) for (const w of this.writesOf(setter)) out.add(w);
					else out.add(attr);
				});
			} else if (node.type === "call") {
				const callee = this.calleeOf(node, { method: m, selfName, chain: [], literals: new Map() });
				if (callee && !callee.deferred) for (const w of this.writesOf(callee)) out.add(w);
				const attr = namedWrite(node, selfName);
				if (attr) out.add(attr);
			}
			for (const child of named(node)) walk(child);
		};
		walk(field(m.node, "body")!);
		return out;
	}

	// The method a call runs on this instance: self.m(), super().m(), or Base.m(self).
	private calleeOf(call: Node, ctx: Context): Method | undefined {
		const fn = field(call, "function");
		if (fn?.type === "identifier") return this.helperOf(call, fn.text, ctx);
		if (fn?.type !== "attribute") return undefined;
		const obj = field(fn, "object")!;
		const name = field(fn, "attribute")!.text;
		if (obj.type === "identifier" && obj.text === ctx.selfName) return this.findMethod(name);
		if (obj.type === "call" && field(obj, "function")?.text === "super") {
			// super(Base, self) starts the lookup after Base, not after the class that holds the call.
			const [start] = callArgs(obj);
			const after = start?.type === "identifier" ? this.mro.find((c) => c.name === start.text) : undefined;
			return this.findMethod(name, after ?? ctx.method.cls);
		}
		if (obj.type === "identifier" && callArgs(call)[0]?.text === ctx.selfName) {
			const cls = this.mro.find((c) => c.name === obj.text);
			if (cls) {
				for (const c of this.mro.slice(this.mro.indexOf(cls))) {
					const m = c.methods.get(name);
					if (m) return m;
				}
			}
			// Other.__init__(self, ...) runs a class that is not a base on this instance.
			const other = this.workspace.resolveClass(ctx.method.file, obj.text).classes[0];
			if (other) for (const c of linearize(other)) if (c.methods.has(name)) return c.methods.get(name);
		}
		return undefined;
	}

	// A module function called with the instance as an argument, e.g. _init_state(self, size).
	private helperOf(call: Node, name: string, ctx: Context): Method | undefined {
		const args = callArgs(call);
		const position = args.findIndex((a) => a.type === "identifier" && a.text === ctx.selfName);
		const keyword = args.find((a) => a.type === "keyword_argument" && field(a, "value")?.text === ctx.selfName);
		if (position < 0 && !keyword) return undefined;
		if (args.slice(0, position < 0 ? args.length : position).some((a) => a.type === "list_splat" || a.type === "dictionary_splat")) return undefined;
		const found = this.workspace.resolveFunction(ctx.method.file, name);
		if (!found) return undefined;
		const params = named(field(found.node, "parameters")!);
		const selfName = keyword ? field(keyword, "name")?.text : paramName(params[position]);
		if (!selfName) return undefined;
		const key = `${found.file}:${found.node.startIndex}:${selfName}`;
		let helper = this.helpers.get(key);
		if (!helper) {
			helper = {
				name,
				cls: ctx.method.cls,
				file: found.file,
				label: `${path.basename(found.file, ".py")}.${name}`,
				node: found.node,
				selfName,
				deferred: isDeferred(found.node),
			};
			this.helpers.set(key, helper);
		}
		return helper;
	}

	private inline(m: Method, state: State, chain: string[], literals = new Map<string, string>()): void {
		if (this.stack.includes(m) || !m.selfName || m.deferred) return;
		// A call with bound literals can assign a different name each time, so it is walked every time.
		if (this.inlined.has(m) && literals.size === 0) {
			for (const w of this.writesOf(m)) state.assigned.add(w);
			return;
		}
		this.inlined.add(m);
		this.stack.push(m);
		const ctx: Context = { method: m, selfName: m.selfName, chain: [...chain, this.label(m)], literals };
		this.visitBlock(field(m.node, "body")!, state, ctx);
		this.stack.pop();
	}

	private visitBlock(block: Node, state: State, ctx: Context): void {
		for (const stmt of named(block)) {
			this.visitStatement(stmt, state, ctx);
			if (stmt.type !== "raise_statement") continue;
			// A raise at a method body's top level means the constructor never returns.
			if (this.branchDepth === 0 && block.id === field(ctx.method.node, "body")?.id) {
				this.markPartial(`${ctx.chain.join(" -> ")} always raises`);
			}
			return;
		}
	}

	private visitStatement(node: Node, state: State, ctx: Context): void {
		const conditional = ["if_statement", "for_statement", "while_statement", "try_statement", "match_statement"].includes(node.type);
		if (conditional) this.branchDepth++;
		try {
			this.visitStatementBody(node, state, ctx);
		} finally {
			if (conditional) this.branchDepth--;
		}
	}

	private visitStatementBody(node: Node, state: State, ctx: Context): void {
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
		const conditional = node.type === "boolean_operator" || node.type === "conditional_expression";
		if (conditional) this.branchDepth++;
		try {
			this.visitExprBody(node, state, ctx);
		} finally {
			if (conditional) this.branchDepth--;
		}
	}

	private visitExprBody(node: Node, state: State, ctx: Context): void {
		switch (node.type) {
			case "lambda":
				return;
			case "assignment": {
				const right = field(node, "right");
				if (!right) return;
				this.visitExpr(right, state, ctx);
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
			const attr = stringLiteral(args[1]) ?? (args[1]?.type === "identifier" ? ctx.literals.get(args[1].text) : undefined);
			if (fn.text === "getattr" && args.length === 2 && attr) this.read(attr, node, state, ctx);
			for (const arg of args.slice(1)) this.visitExpr(arg, state, ctx);
			if (fn.text === "setattr" && attr) state.assigned.add(attr);
			else if (fn.text === "setattr") this.markPartial(`${ctx.chain.join(" -> ")} calls setattr with a computed name`);
			return;
		}
		const callee = this.calleeOf(node, ctx);
		if (callee) {
			for (const arg of args) this.visitExpr(arg, state, ctx);
			this.inline(callee, state, ctx.chain, this.bindLiterals(fn, callee, args, ctx));
			return;
		}
		if (this.opensUnknownScope(fn, args, ctx)) this.markPartial(`${ctx.chain.join(" -> ")} calls ${fn.text} past the known MRO`);
		this.visitExpr(fn, state, ctx);
		for (const arg of args) this.visitExpr(arg, state, ctx);
		const registered = namedWrite(node, ctx.selfName);
		if (registered) state.assigned.add(registered);
	}

	// For self.m("x", ...), the parameters of m that receive a string literal.
	private bindLiterals(fn: Node, callee: Method, args: Node[], ctx: Context): Map<string, string> {
		const out = new Map<string, string>();
		if (fn.type !== "attribute" || field(fn, "object")?.text !== ctx.selfName) return out;
		const params = named(field(callee.node, "parameters")!).slice(1).map(paramName);
		args.forEach((arg, i) => {
			const value = arg.type === "keyword_argument" ? field(arg, "value") : arg;
			const name = arg.type === "keyword_argument" ? field(arg, "name")?.text : params[i];
			const literal = stringLiteral(value ?? undefined) ?? (value?.type === "identifier" ? ctx.literals.get(value.text) : undefined);
			if (name && literal) out.set(name, literal);
		});
		return out;
	}

	// A super() call with no target in the known MRO goes to whatever class follows a mixin at runtime.
	private opensUnknownScope(fn: Node, args: Node[], ctx: Context): boolean {
		if (fn.type !== "attribute") return false;
		const obj = field(fn, "object")!;
		if (obj.type === "call" && field(obj, "function")?.text === "super") {
			// With no base outside the repo, an unresolved super() call only makes sense in a mixin.
			const mixin = this.mro.every((c) => c.unresolvedBases.length === 0);
			const unresolved = this.findMethod(field(fn, "attribute")!.text, ctx.method.cls) === undefined;
			return (mixin && unresolved) || args.some((a) => a.type === "list_splat" || a.type === "dictionary_splat");
		}
		return obj.text === `${ctx.selfName}.__dict__` && field(fn, "attribute")!.text === "update";
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
		if (this.dynamic || state.assigned.has(attr) || state.guarded.has(attr)) return;
		const getter = this.findGetter(attr);
		if (getter) {
			this.inline(getter, state, ctx.chain);
			return;
		}
		if (this.opaque || this.isClassLevel(attr) || !this.known.has(attr)) return;
		const file = ctx.method.file;
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

export async function createWorkspace(packageRoot: string): Promise<Workspace> {
	return new Workspace(await getParser(), packageRoot);
}

export type { Workspace };

// Reports undefined when the class has no __init__ inside the repo, e.g. one generated by msgspec.
export function analyzeClass(workspace: Workspace, file: string, className: string): Report | undefined {
	const cls = workspace.resolveClass(file, className).classes.find((c) => c.file === file);
	if (!cls) throw new Error(`${file}: no class ${className}`);
	const simulator = new Simulator(cls, workspace);
	if (!simulator.hasInit()) return undefined;
	return simulator.run();
}

// Top-level class names defined in a module.
export function classNames(workspace: Workspace, file: string): string[] {
	return workspace.definedClasses(file);
}

export async function analyze(repoRoot: string, packageRoot: string, target: Target): Promise<Report> {
	const workspace = await createWorkspace(packageRoot);
	const report = analyzeClass(workspace, path.join(repoRoot, target.file), target.className);
	if (!report) throw new Error(`${target.file}: ${target.className} has no __init__ in its resolved MRO`);
	return report;
}
