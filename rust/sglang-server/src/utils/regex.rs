//! `stop_regex` validation and bounding.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use super::error::Error;

/// `MAX_LEN` from Python's `get_max_seq_length`: the bound for an *unbounded* stop regex (`\d+`, `.*`, …) or one we can't statically size.
const STOP_REGEX_MAX_LEN: usize = 1 << 30;

/// Escapes that mean the same thing to `regex-syntax` and to Python's `re`. An allowlist, not a blocklist.
const PORTABLE_FLAGS: &[char] = &['i', 'm', 's', 'x', 'u', '-'];

/// Cap on the PRODUCT of counted repeats along one path.
const MAX_REPEAT_COUNT: u64 = 512;

/// Limit on [`ambiguity_degree`].
const MAX_AMBIGUITY_DEGREE: u64 = 1;

const REGEX_AST_NEST_LIMIT: u32 = 64;

const SHARED_ESCAPES: &[char] = &[
    'A', 'b', 'B', 'd', 'D', 's', 'S', 'w', 'W', 'a', 'f', 'n', 'r', 't', 'v',
    // `\xHH` (exactly hex digits) is shared.
    'x',
];

/// Reject the constructs `regex-syntax` accepts but Python's `re` cannot compile.
///
/// Everything else in this module rests on one property: **anything Rust admits,
/// Python can compile.** The reverse is allowed to fail — rejecting a pattern
/// `re.search` runs on the decode hot path where nothing catches it.
fn reject_python_incompatible(pattern: &str) -> Result<(), Error> {
    let reject = |what: String| {
        Err(Error::Validation(format!(
            "stop_regex {pattern:?} uses {what}, which Python's `re` cannot compile"
        )))
    };
    // ASCII-only comparisons, so scanning bytes is safe: a UTF-8 continuation byte is >= 0x80 and matches no arm.
    let b = pattern.as_bytes();
    let mut i = 0;
    // Still inside the run of leading `(?flags)` groups.
    let mut leading = true;
    while i < b.len() {
        match b[i] {
            b'\\' => {
                if let Err(what) = check_escape(b, i) {
                    return reject(what);
                }
                leading = false;
                i += 2; // skip the escaped character, so `\(` is not a group open
            }
            // `(?<name>…)` is a named group to Rust; Python spells it
            // `(?P<name>…)` and errors on this.
            b'(' if b[i..].starts_with(b"(?<")
                && !b[i..].starts_with(b"(?<=")
                && !b[i..].starts_with(b"(?<!") =>
            {
                return reject("a `(?<name>…)` group (Python spells it `(?P<name>…)`)".into());
            }
            // A flag-setting group. The flag letters also differ: Rust adds `R`/`U`,
            // Python adds `a`/`L`, so only their intersection is portable.
            b'(' if flag_group_bytes(&b[i..]).is_some() => {
                let flags = flag_group_bytes(&b[i..]).expect("just matched");
                // `(?flags:…)` is scoped: legal anywhere, and its clearing form is legal too.
                let scoped = b[i..].get(2 + flags.len()).is_some_and(|&c| c == b':');
                // Python allows global flags only at the start, but allows SEVERAL
                // (`(?i)(?m)a`); `leading` stays true while we are still in that run.
                if !scoped && !leading {
                    return reject("inline flags after the start of the pattern".into());
                }
                if !scoped && flags.contains(&b'-') {
                    return reject(
                        "a clearing `(?-flags)` group (Python wants `(?-flags:…)`)".into(),
                    );
                }
                if let Some(&f) = flags
                    .iter()
                    .find(|f| !PORTABLE_FLAGS.contains(&(**f as char)))
                {
                    return reject(format!("the inline flag `{}`", f as char));
                }
                // Advance past the WHOLE group, not one byte.
                if scoped {
                    leading = false;
                    i += 1;
                } else {
                    i += 2 + flags.len() + 1; // `(?` + flags + `)`
                }
                continue; // still in the leading flag run
            }
            // A `[` inside a character class. Rust reads it as a literal (or a POSIX
            // class); Python's parser terminates the class differently and can end up
            // parsing the remainder as a group.
            b'[' => {
                let mut j = i + 1;
                if b.get(j) == Some(&b'^') {
                    j += 1;
                }
                if b.get(j) == Some(&b']') {
                    j += 1; // a leading `]` is a literal in both dialects
                }
                while j < b.len() && b[j] != b']' {
                    match b[j] {
                        // Escapes inside a class follow the same rules as outside.
                        b'\\' => {
                            if let Err(what) = check_escape(b, j) {
                                return reject(what);
                            }
                            j += 2;
                        }
                        b'[' => return reject("a `[` nested inside a character class".into()),
                        // `[a--b]` is a class-difference operator in Rust and a bad
                        // character range in Python.
                        b'-' if b.get(j + 1) == Some(&b'-') => {
                            return reject("a `--` class-difference operator".into());
                        }
                        _ => j += 1,
                    }
                }
                i = j.max(i + 1);
            }
            _ => {
                leading = false;
                i += 1;
            }
        }
    }
    Ok(())
}

/// The flag bytes of a flag-setting group (`(?i)`, `(?-i)`, `(?imsx)`), or `None`
/// if `b` does not open one. A `(?i:…)` scoped group is not one of these.
fn flag_group_bytes(b: &[u8]) -> Option<&[u8]> {
    let rest = b.strip_prefix(b"(?")?;
    // Stop.
    let end = rest.iter().position(|&c| c == b')' || c == b':')?;
    let flags = &rest[..end];
    (!flags.is_empty() && flags.iter().all(|&c| c.is_ascii_alphabetic() || c == b'-'))
        .then_some(flags)
}

/// Check the escape starting at `b[i]` (a backslash). `Err` names why Python's
/// `re` would refuse it. Used for escapes both inside and outside character
/// classes — the class scanner used to skip escapes entirely, which is how
/// `[\p{L}]` slipped past the very check written for `\p{L}`.
fn check_escape(b: &[u8], i: usize) -> Result<(), String> {
    let Some(&e) = b.get(i + 1) else {
        return Err("a trailing backslash".into());
    };
    // `\xHH` is shared; Rust's braced `\x{10FFFF}` is not.
    if e == b'x' && b.get(i + 2) == Some(&b'{') {
        return Err("a braced `\\x{…}` escape".into());
    }
    // `\b{start}` is one zero-width assertion to Rust, but `\b` followed by
    // the literal "{start}" to Python.
    if e == b'b' && b.get(i + 2) == Some(&b'{') {
        return Err("a `\\b{…}` assertion".into());
    }
    if e.is_ascii_alphanumeric() && !SHARED_ESCAPES.contains(&(e as char)) {
        return Err(format!("the escape `\\{}`", e as char));
    }
    if e == b'<' || e == b'>' {
        return Err(format!("the escape `\\{}`", e as char));
    }
    Ok(())
}

/// Entries kept in [`ADMISSION_CACHE`], mirroring CPython's `re._MAXCACHE`.
const ADMISSION_CACHE_CAP: usize = 512;

/// Memo of admitted patterns → their bound.
static ADMISSION_CACHE: LazyLock<Mutex<HashMap<Box<str>, usize>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn cached_bound(pattern: &str) -> Option<usize> {
    ADMISSION_CACHE
        .lock()
        .ok()
        .and_then(|c| c.get(pattern).copied())
}

fn cache_bound(pattern: &str, max_len: usize) {
    let Ok(mut c) = ADMISSION_CACHE.lock() else {
        return;
    };
    if c.len() >= ADMISSION_CACHE_CAP {
        c.clear();
    }
    c.insert(pattern.into(), max_len);
}

/// A `stop_regex` that has been admitted, together with the bound derived
/// while admitting it.
pub struct RegexPattern<'a> {
    pattern: &'a str,
    max_len: usize,
}

/// `TryFrom`, not `FromStr`: `FromStr::from_str` takes a `&str` whose
/// lifetime the trait never names.
impl<'a> TryFrom<&'a str> for RegexPattern<'a> {
    type Error = Error;

    fn try_from(pattern: &'a str) -> Result<Self, Self::Error> {
        Self::build(pattern)
    }
}

impl<'a> RegexPattern<'a> {
    /// Admit `pattern` and derive its bound in a single AST walk.
    ///
    /// `Err` for anything CPython's `re` cannot compile, or cannot match cheaply
    /// enough to run on every decode step — see [`validate`].
    fn build(pattern: &'a str) -> Result<Self, Error> {
        // Same pattern text ⇒ same verdict and same bound, so a repeat costs a hash
        // lookup instead of a parse + translate. See [`ADMISSION_CACHE`].
        if let Some(max_len) = cached_bound(pattern) {
            return Ok(Self { pattern, max_len });
        }
        let ast = validate(pattern)?;
        // Translate the AST `validate` already produced instead of
        // re-parsing.
        let hir = regex_syntax::hir::translate::TranslatorBuilder::new()
            .build()
            .translate(pattern, &ast)
            .map_err(|e| {
                Error::Validation(format!(
                    "stop_regex {pattern:?} is not a valid regular expression: {e}"
                ))
            })?;
        let max_len = hir_max_len(&hir);
        cache_bound(pattern, max_len);
        Ok(Self { pattern, max_len })
    }

    /// The admitted pattern. See the field note on why this is kept.
    #[allow(dead_code)]
    pub fn pattern(&self) -> &str {
        self.pattern
    }

    pub fn max_len(&self) -> usize {
        self.max_len
    }
}

/// Validate a `stop_regex` before it can reach the scheduler, returning the parsed
/// AST so the caller can derive its bound without parsing again.
fn validate(pattern: &str) -> Result<regex_syntax::ast::Ast, Error> {
    reject_python_incompatible(pattern)?;
    let ast = regex_syntax::ast::parse::ParserBuilder::new()
        .nest_limit(REGEX_AST_NEST_LIMIT)
        .build()
        .parse(pattern)
        .map_err(|e| {
            Error::Validation(format!(
                "stop_regex {pattern:?} is not a valid regular expression: {e}"
            ))
        })?;
    if repetition_cost_too_large(&ast, 1, false) {
        return Err(Error::Validation(format!(
            "stop_regex {pattern:?} repeats too many times or nests unbounded \
             repetitions; matching it would dominate every decode step"
        )));
    }
    if repeats_an_assertion(&ast) {
        return Err(Error::Validation(format!(
            "stop_regex {pattern:?} quantifies a zero-width assertion, which Python's \
             `re` rejects, or a repetition count Python cannot honour"
        )));
    }
    if alternation_under_repetition(&ast) {
        return Err(Error::Validation(format!(
            "stop_regex {pattern:?} alternates inside a repetition; each iteration \
             could match more than one way, so Python's backtracking engine explores \
             exponentially many parses"
        )));
    }
    match ambiguity_degree(&ast) {
        None => {
            return Err(Error::Validation(format!(
                "stop_regex {pattern:?} repeats a variable-length expression without \
                 a bound; matching it would dominate every decode step"
            )));
        }
        Some(d) if d > MAX_AMBIGUITY_DEGREE => {
            return Err(Error::Validation(format!(
                "stop_regex {pattern:?} has {d} independent length choices (limit \
                 {MAX_AMBIGUITY_DEGREE}); Python's backtracking engine would explore \
                 their product on every decode step"
            )));
        }
        Some(_) => {}
    }
    Ok(ast)
}

/// Reject repetitions whose cost compounds down the nesting.
///
/// `outer` is the product of the counted repeats enclosing `ast`. Families die
/// here: a counted product over [`MAX_REPEAT_COUNT`] (memory), and an unbounded
/// tail grows every step, so the loop is dead a bounded number of tokens).
fn repetition_cost_too_large(ast: &regex_syntax::ast::Ast, outer: u64, unbounded: bool) -> bool {
    use regex_syntax::ast::{Ast, RepetitionKind, RepetitionRange};
    match ast {
        Ast::Repetition(rep) => {
            let (factor, is_unbounded) = match &rep.op.kind {
                RepetitionKind::Range(RepetitionRange::Exactly(n)) => (*n as u64, false),
                RepetitionKind::Range(RepetitionRange::Bounded(_, hi)) => (*hi as u64, false),
                RepetitionKind::Range(RepetitionRange::AtLeast(n)) => (*n as u64, true), // codespell:ignore atleast
                _ => (1, true), // `*`, `+`, `?`
            };
            let total = outer.saturating_mul(factor.max(1));
            total >= MAX_REPEAT_COUNT
                || (is_unbounded && unbounded)
                || repetition_cost_too_large(&rep.ast, total, unbounded || is_unbounded)
        }
        Ast::Group(g) => repetition_cost_too_large(&g.ast, outer, unbounded),
        Ast::Concat(c) => c
            .asts
            .iter()
            .any(|a| repetition_cost_too_large(a, outer, unbounded)),
        Ast::Alternation(a) => a
            .asts
            .iter()
            .any(|a| repetition_cost_too_large(a, outer, unbounded)),
        _ => false,
    }
}

/// Whether any repetition in `ast` applies to a zero-width assertion — `$*`,
/// `\b{2}`, `^+`. `regex-syntax` accepts them.
/// repeat". Found by fuzzing both parsers against each other, not by reading
/// either one's docs.
///
/// Checked on the AST, not the HIR: the HIR translator folds `$+` down to a bare
/// `Look`, so by then the shape Python objects to is gone.
fn repeats_an_assertion(ast: &regex_syntax::ast::Ast) -> bool {
    use regex_syntax::ast::Ast;
    match ast {
        // A quantified assertion (`$*`) or a quantified quantifier (`a?*`, which
        // Python calls "multiple repeat"). Both parse fine in Rust.
        Ast::Repetition(rep) => {
            matches!(&*rep.ast, Ast::Assertion(_) | Ast::Repetition(_))
                || repeats_an_assertion(&rep.ast)
        }
        Ast::Group(g) => repeats_an_assertion(&g.ast),
        Ast::Concat(c) => c.asts.iter().any(repeats_an_assertion),
        Ast::Alternation(a) => a.asts.iter().any(repeats_an_assertion),
        _ => false,
    }
}

/// How many independent length choices a backtracking engine must enumerate. `None` means unbounded (exponential). This is the predicate review rounds of structural rules kept missing, and it is the only one whose threshold was chosen by MEASUREMENT rather than argument. Only composition trips the limit — several in a row (`a*a*a*b`), or one over a body that is itself variable-length (`(?:a*){10}`). The check is also orthogonal to the returned bound: [`hir_max_len`] is untouched, so admitting a pattern never changes the window the scheduler sizes for it.
fn ambiguity_degree(ast: &regex_syntax::ast::Ast) -> Option<u64> {
    use regex_syntax::ast::{Ast, RepetitionKind, RepetitionRange};
    match ast {
        Ast::Group(g) => ambiguity_degree(&g.ast),
        // Siblings compose: `a*a*a*b` is independent choices, and every one
        // multiplies the work.
        Ast::Concat(c) => c.asts.iter().try_fold(0u64, |acc, a| {
            Some(acc.saturating_add(ambiguity_degree(a)?))
        }),
        Ast::Alternation(a) => a
            .asts
            .iter()
            .try_fold(0u64, |acc, x| Some(acc.max(ambiguity_degree(x)?))),
        Ast::Repetition(rep) => {
            let body = ambiguity_degree(&rep.ast)?;
            let (lo, hi) = match &rep.op.kind {
                RepetitionKind::ZeroOrOne => (0u64, Some(1u64)),
                RepetitionKind::ZeroOrMore => (0, None),
                RepetitionKind::OneOrMore => (1, None),
                RepetitionKind::Range(RepetitionRange::Exactly(n)) => (*n as u64, Some(*n as u64)),
                RepetitionKind::Range(RepetitionRange::AtLeast(n)) => (*n as u64, None), // codespell:ignore atleast
                RepetitionKind::Range(RepetitionRange::Bounded(a, b)) => {
                    (*a as u64, Some(*b as u64))
                }
            };
            match hi {
                // Unbounded. Repeating an unambiguous fixed-length body is one
                // choice (`a*`, `(?:ab)*`); repeating anything else is exponential.
                None => {
                    if body > 0 || is_variable_length(&rep.ast) {
                        None
                    } else {
                        Some(1)
                    }
                }
                // Counted: the body's own freedom is paid once per iteration.
                Some(hi) => Some(hi.saturating_mul(body).saturating_add(u64::from(lo != hi))),
            }
        }
        _ => Some(0),
    }
}

/// Whether any alternation sits inside a repetition body.
///
/// parses. A top-level alternation (`and|or`.
/// untouched: only a repetition of one is refused.
fn alternation_under_repetition(ast: &regex_syntax::ast::Ast) -> bool {
    use regex_syntax::ast::Ast;
    fn contains_alternation(ast: &Ast) -> bool {
        match ast {
            Ast::Alternation(_) => true,
            Ast::Group(g) => contains_alternation(&g.ast),
            Ast::Concat(c) => c.asts.iter().any(contains_alternation),
            Ast::Repetition(r) => contains_alternation(&r.ast),
            _ => false,
        }
    }
    match ast {
        Ast::Repetition(rep) => {
            contains_alternation(&rep.ast) || alternation_under_repetition(&rep.ast)
        }
        Ast::Group(g) => alternation_under_repetition(&g.ast),
        Ast::Concat(c) => c.asts.iter().any(alternation_under_repetition),
        Ast::Alternation(a) => a.asts.iter().any(alternation_under_repetition),
        _ => false,
    }
}

/// Strict upper bound on the characters `hir` can match; `None` (unbounded) maps to
/// the full-scan sentinel. Saturating throughout: a nested `{65535}` repeat would
/// otherwise overflow into a small — and therefore unsafe — bound.
fn hir_max_len(hir: &regex_syntax::hir::Hir) -> usize {
    use regex_syntax::hir::HirKind;
    match hir.kind() {
        HirKind::Empty | HirKind::Look(_) => 0,
        HirKind::Literal(lit) => lit.0.len(),
        HirKind::Class(_) => 1,
        HirKind::Repetition(rep) => match rep.max {
            None => STOP_REGEX_MAX_LEN,
            Some(max) => (max as usize)
                .saturating_mul(hir_max_len(&rep.sub))
                .min(STOP_REGEX_MAX_LEN),
        },
        HirKind::Capture(cap) => hir_max_len(&cap.sub),
        HirKind::Concat(subs) => subs
            .iter()
            .map(hir_max_len)
            .fold(0, usize::saturating_add)
            .min(STOP_REGEX_MAX_LEN),
        HirKind::Alternation(subs) => subs.iter().map(hir_max_len).max().unwrap_or(0),
    }
}

/// Whether `ast` can match more than one length — the property that makes a
/// repetition of it ambiguous.
fn is_variable_length(ast: &regex_syntax::ast::Ast) -> bool {
    let (lo, hi) = ast_len(ast);
    hi != Some(lo)
}

/// Saturating `(min, max)` match length of `ast`; `max = None` means unbounded.
///
/// Deliberately on the AST rather than the HIR.
/// a single class and `$+` into a bare `Look`, erasing exactly the shapes CPython's
/// engine still has to enumerate.
fn ast_len(ast: &regex_syntax::ast::Ast) -> (u64, Option<u64>) {
    use regex_syntax::ast::{Ast, RepetitionKind, RepetitionRange};
    match ast {
        Ast::Empty(_) | Ast::Flags(_) | Ast::Assertion(_) => (0, Some(0)),
        Ast::Literal(_) | Ast::Dot(_) | Ast::ClassUnicode(_) | Ast::ClassPerl(_) => (1, Some(1)),
        Ast::ClassBracketed(_) => (1, Some(1)),
        Ast::Group(g) => ast_len(&g.ast),
        Ast::Concat(c) => c.asts.iter().fold((0, Some(0)), |(lo, hi), a| {
            let (l, h) = ast_len(a);
            (
                lo.saturating_add(l),
                match (hi, h) {
                    (Some(x), Some(y)) => Some(x.saturating_add(y)),
                    _ => None,
                },
            )
        }),
        Ast::Alternation(a) => a.asts.iter().fold((u64::MAX, Some(0)), |(lo, hi), x| {
            let (l, h) = ast_len(x);
            (
                lo.min(l),
                match (hi, h) {
                    (Some(p), Some(q)) => Some(p.max(q)),
                    _ => None,
                },
            )
        }),
        Ast::Repetition(rep) => {
            let (l, h) = ast_len(&rep.ast);
            let (lo, hi) = match &rep.op.kind {
                RepetitionKind::ZeroOrOne => (0u64, Some(1u64)),
                RepetitionKind::ZeroOrMore => (0, None),
                RepetitionKind::OneOrMore => (1, None),
                RepetitionKind::Range(RepetitionRange::Exactly(n)) => (*n as u64, Some(*n as u64)),
                RepetitionKind::Range(RepetitionRange::AtLeast(n)) => (*n as u64, None), // codespell:ignore atleast
                RepetitionKind::Range(RepetitionRange::Bounded(a, b)) => {
                    (*a as u64, Some(*b as u64))
                }
            };
            (
                lo.saturating_mul(l),
                match (hi, h) {
                    (Some(x), Some(y)) => Some(x.saturating_mul(y)),
                    _ => None,
                },
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bound-only view of [`RegexPattern`], so the corpus rows read as
    /// `pattern -> bound` without naming the type at every call.
    fn stop_regex_bound(pattern: &str) -> Result<usize, Error> {
        RegexPattern::try_from(pattern).map(|r| r.max_len())
    }

    /// The admission memo must be indistinguishable from admitting afresh.
    #[test]
    fn admission_memo_agrees_with_admitting_afresh() {
        // Distinct from any other test's patterns: the cache is process-wide.
        let admitted = r"memo\d{3}[a-f]+";
        let cold = RegexPattern::try_from(admitted).expect("valid").max_len();
        let warm = RegexPattern::try_from(admitted).expect("valid").max_len();
        assert_eq!(
            cold, warm,
            "a memoized bound must equal a freshly derived one"
        );

        // Rejections are re-validated every time, so the memo can never turn one into an admission.
        let rejected = r"memo(?:.|.)*Z";
        assert!(RegexPattern::try_from(rejected).is_err());
        assert!(
            RegexPattern::try_from(rejected).is_err(),
            "a rejected pattern must stay rejected on the second try"
        );

        // Overflow the cache, then re-check: clearing must not corrupt or stale a
        // subsequent lookup.
        for i in 0..=ADMISSION_CACHE_CAP {
            let _ = RegexPattern::try_from(format!("memofill{i}").as_str());
        }
        assert_eq!(
            RegexPattern::try_from(admitted).expect("valid").max_len(),
            cold,
            "the bound must survive a cache clear"
        );
    }

    #[test]
    fn admitted_pattern_carries_its_own_text_and_bound() {
        let p = RegexPattern::try_from(r"\d{6}").expect("valid");
        assert_eq!(p.pattern(), r"\d{6}");
        assert_eq!(p.max_len(), 6);
    }

    /// The property this whole design rests on: **anything Rust admits, Python can compile.**
    const SEARCH_BUDGET_MS: f64 = 5.0;

    /// What the admission policy must do with a pattern.
    #[derive(Debug, PartialEq)]
    enum Policy {
        /// Admitting it kills the scheduler or silently misses the stop.
        MustReject,
        /// Admitting it is REQUIRED.
        MustAdmit,
        /// Python compiles it; Rust may or may not, and either verdict passes.
        MayReject,
    }

    /// One corpus row.
    struct Case {
        pattern: String,
        policy: Policy,
        /// Expected bound when admitted. Pins `hir_max_len` against silent drift.
        rust_bound: usize,
        py_max_len: Option<i64>,
        worst_ms: f64,
    }

    fn case(pattern: &str, policy: Policy, rust_bound: usize, py: Option<i64>, ms: f64) -> Case {
        Case {
            pattern: pattern.to_string(),
            policy,
            rust_bound,
            py_max_len: py,
            worst_ms: ms,
        }
    }

    /// The source of truth for `stop_regex` admission. The contract is ONE-SIDED: the admitted set must be a SUBSET of what CPython can compile and match cheaply. Rust does not reproduce Python's dialect — rejecting a pattern Python accepts costs the client a feature, admitting one Python chokes on costs the scheduler and the GPU state. So `MustReject` carries the whole safety burden, and `MustAdmit` is held to the few patterns the project's own tests send. This table exists because review rounds each found a NEW spelling of an already-fixed hazard, and the corpus could not catch any of them: its assertion was `!admitted || python_compiles`, which any row with `python_compiles = true` satisfies vacuously — including a few rows whose own comments called them scheduler-fatal. KEEP IN SYNC: adding a row means MEASURING `py_max_len` and `worst_ms`, not guessing them. `corpus_rows_are_self_consistent` refuses a row that records a fatal measurement and then claims the pattern is safe to admit.
    fn corpus() -> Vec<Case> {
        const UNBOUNDED: usize = STOP_REGEX_MAX_LEN;
        const INF: f64 = f64::INFINITY;
        let mut c = vec![
            // ---- Direction A: CPython cannot compile these.
            case(r"\p{L}", Policy::MustReject, 0, None, INF),
            case(r"\P{L}", Policy::MustReject, 0, None, INF),
            case(r"\pL", Policy::MustReject, 0, None, INF),
            case("(?<n>a)", Policy::MustReject, 0, None, INF),
            case(r"\x{1F600}", Policy::MustReject, 0, None, INF),
            case(r"\u{41}", Policy::MustReject, 0, None, INF),
            case("(?<=a*)b", Policy::MustReject, 0, Some(1073741825), INF),
            case("(", Policy::MustReject, 0, None, INF),
            case("[z-a]", Policy::MustReject, 0, None, INF),
            case("a{2,1}", Policy::MustReject, 0, None, INF),
            case("$*", Policy::MustReject, 0, None, INF),
            case(r"\b{2}", Policy::MustReject, 0, None, INF),
            case("^+", Policy::MustReject, 0, None, INF),
            case("a?*", Policy::MustReject, 0, None, INF),
            case("a{2,5}?*", Policy::MustReject, 0, None, INF),
            case("a(?i)b", Policy::MustReject, 0, None, INF),
            case("(?-i)a", Policy::MustReject, 0, None, INF),
            case("[a[:alpha:](?=-]", Policy::MustReject, 0, None, INF),
            case(r"[\p{L}]", Policy::MustReject, 0, None, INF),
            case(r"[\pL]", Policy::MustReject, 0, None, INF),
            case(r"[\P{L}]", Policy::MustReject, 0, None, INF),
            case(r"[\x{41}]", Policy::MustReject, 0, None, INF),
            case("[a--b]", Policy::MustReject, 0, None, INF),
            case("(?R)a", Policy::MustReject, 0, None, INF),
            case("(?U)a", Policy::MustReject, 0, None, INF),
            case("(?R:a)", Policy::MustReject, 0, None, INF),
            case("(?U:a)", Policy::MustReject, 0, None, INF),
            // `regex-syntax` parses counts as u32 and accepts up to u32::MAX.
            case("a{4294967295}", Policy::MustReject, 0, None, INF),
            case("a{5000000000}", Policy::MustReject, 0, None, INF),
            // ---- Bound UNDER-estimates.
            case(r"\<END\>", Policy::MustReject, 3, Some(5), 0.02),
            case(r"\b{start}xyz", Policy::MustReject, 3, Some(10), 0.04),
            // ---- Compounding repeat cost. Both compile in CPython; both are fatal there.
            case(
                "(?:(?:a*){65535}){65535}",
                Policy::MustReject,
                0,
                Some(4611545282012774400),
                INF,
            ),
            case("(?:){1048575}x", Policy::MustReject, 0, Some(1), INF),
            case(
                "(?:.|.)*Z",
                Policy::MustReject,
                UNBOUNDED,
                Some(1073741825),
                INF,
            ),
            case(
                "(a|a)*b",
                Policy::MustReject,
                UNBOUNDED,
                Some(1073741825),
                INF,
            ),
            case("(?:a+)+b", Policy::MustReject, 0, Some(1073741825), INF),
            case(
                "(?:a*){10}b",
                Policy::MustReject,
                UNBOUNDED,
                Some(10737418241),
                INF,
            ),
            case(
                "a*a*a*a*a*a*a*a*b",
                Policy::MustReject,
                UNBOUNDED,
                Some(8589934593),
                636.05,
            ),
            case(
                "(?:.*){20}Z",
                Policy::MustReject,
                UNBOUNDED,
                Some(21474836481),
                INF,
            ),
            case(
                ".*.*.*.*.*.*.*.*Z",
                Policy::MustReject,
                UNBOUNDED,
                Some(8589934593),
                INF,
            ),
            case("(?:.?){30}Z", Policy::MustReject, 31, Some(31), INF),
            case("(?:.?){255}Z", Policy::MustReject, 256, Some(256), INF),
            case(
                "(?:.{0,1}.{0,1}.{0,1}){8}Z",
                Policy::MustReject,
                25,
                Some(25),
                INF,
            ),
            case(
                "(?:(?:.?){15}){15}Z",
                Policy::MustReject,
                226,
                Some(226),
                INF,
            ),
            case("(?:.?.?.?.?){60}Z", Policy::MustReject, 241, Some(241), INF),
            // ---- MustAdmit.
            case(
                r"[.!?]\s*$",
                Policy::MustAdmit,
                UNBOUNDED,
                Some(1073741825),
                0.03,
            ),
            case("and|or", Policy::MustAdmit, 3, Some(3), 0.03),
            case(r"\d+", Policy::MayReject, UNBOUNDED, Some(1073741824), 0.03),
            case(
                r"\s+$",
                Policy::MayReject,
                UNBOUNDED,
                Some(1073741824),
                0.04,
            ),
            case(
                "Answer: .*",
                Policy::MayReject,
                UNBOUNDED,
                Some(1073741832),
                0.03,
            ),
            case(".*", Policy::MayReject, UNBOUNDED, Some(1073741824), 0.04),
            case(
                "a{3,}",
                Policy::MayReject,
                UNBOUNDED,
                Some(1073741824),
                0.03,
            ),
            case("colou?r", Policy::MustAdmit, 6, Some(6), 0.03),
            case("https?://", Policy::MayReject, 8, Some(8), 0.02),
            case("END(ING)?", Policy::MayReject, 6, Some(6), 0.02),
            case(
                "(?i)[a-z]+",
                Policy::MayReject,
                UNBOUNDED,
                Some(1073741824),
                0.04,
            ),
            case(r"(?i)\d{4}-\d{2}", Policy::MayReject, 7, Some(7), 0.06),
            case("(?imsx)a-b", Policy::MayReject, 3, Some(3), 0.04),
            case("(?-i:abc)", Policy::MayReject, 3, Some(3), 0.03),
            case("(?i-s:a)", Policy::MayReject, 1, Some(1), 0.04),
            case("(?i)(?m)a", Policy::MayReject, 1, Some(1), 0.03),
            case(r"\x41", Policy::MayReject, 1, Some(1), 0.02),
            case(r"\d{6}", Policy::MustAdmit, 6, Some(6), 0.03),
            case("abc", Policy::MustAdmit, 3, Some(3), 0.03),
            case("(?P<n>a)", Policy::MayReject, 1, Some(1), 0.03),
            case(r"a\.b", Policy::MayReject, 3, Some(3), 0.03),
            case(r"\bword\b", Policy::MayReject, 4, Some(4), 0.03),
            case(r"[\d\s]{2}", Policy::MayReject, 2, Some(2), 0.03),
            // ---- MayReject: CPython accepts, `regex-syntax` is stricter.
            case(r"a\Z", Policy::MayReject, 1, Some(1), 0.03),
            case(r"(a)\1", Policy::MayReject, 0, Some(1073741825), 0.04),
            case("(?=x)y", Policy::MayReject, 0, Some(1073741825), 0.03),
            case("a{,5}", Policy::MayReject, 5, Some(5), 0.04),
            case(r"\N{SNOWMAN}", Policy::MayReject, 1, Some(1), 0.03),
            case(r"\0", Policy::MayReject, 1, Some(1), 0.03),
        ];
        // Flat concatenations of optional atoms — the round-8 escape.
        c.push(case(
            &format!("{}Z", ".{0,1}".repeat(20)),
            Policy::MustReject,
            21,
            Some(21),
            650.62,
        ));
        c.push(case(
            &format!("{}Z", ".{0,4}".repeat(12)),
            Policy::MustReject,
            49,
            Some(49),
            INF,
        ));
        c
    }

    /// A row may not record a fatal measurement and then claim the pattern is safe to admit.
    #[test]
    fn corpus_rows_are_self_consistent() {
        for c in corpus() {
            if c.py_max_len.is_none() || c.worst_ms > SEARCH_BUDGET_MS {
                assert_eq!(
                    c.policy,
                    Policy::MustReject,
                    "{:?} does not compile in Python, or costs {} ms per decode step \
                     (budget {SEARCH_BUDGET_MS} ms) — it cannot be admitted",
                    c.pattern,
                    c.worst_ms
                );
            }
        }
    }

    /// The corpus, asserted in BOTH directions plus the bound.
    #[test]
    fn stop_regex_corpus_holds_in_both_directions() {
        let mut failures: Vec<String> = Vec::new();
        for c in corpus() {
            let got = stop_regex_bound(&c.pattern);
            match (&c.policy, &got) {
                (Policy::MustReject, Ok(bound)) => failures.push(format!(
                    "ADMITTED but must be rejected: {:?} (bound {bound}, \
                     worst re.search {} ms)",
                    c.pattern, c.worst_ms
                )),
                (Policy::MustAdmit, Err(e)) => failures.push(format!(
                    "REJECTED but must be admitted: {:?} — {e}",
                    c.pattern
                )),
                _ => {}
            }
            if let Ok(bound) = got {
                if c.policy != Policy::MustReject && bound != c.rust_bound {
                    failures.push(format!(
                        "bound drift: {:?} expected {} got {bound}",
                        c.pattern, c.rust_bound
                    ));
                }
                // Only meaningful when CPython's own bound is finite: for unbounded
                // patterns both sides emit an absurd sentinel that the scheduler
                // caps at the output length anyway.
                if let Some(py) = c.py_max_len
                    && py < STOP_REGEX_MAX_LEN as i64
                    && (bound as i64) < py
                {
                    {
                        failures.push(format!(
                            "UNDER-estimate: {:?} rust bound {bound} < python {py} — \
                             the scheduler's window is too small and the stop never fires",
                            c.pattern
                        ));
                    }
                }
            }
        }
        assert!(
            failures.is_empty(),
            "{} corpus row(s) failed:\n  {}",
            failures.len(),
            failures.join("\n  ")
        );
    }

    /// The leading-flag check must look at the FLAG BYTES, not the rest of the pattern.
    #[test]
    fn leading_inline_flags_are_accepted() {
        for pattern in [
            "(?i)[a-z]{1,8}",
            r"(?i)\d{4}-\d{2}",
            "(?imsx)a-b",
            "(?i)abc",
        ] {
            assert!(
                stop_regex_bound(pattern).is_ok(),
                "{pattern} is valid Python and must not be rejected"
            );
        }
        // …but only leading, only set-flags, and only portable letters.
        for pattern in ["a(?i)b", "(?-i)a", "(?R)a", "(?U)a"] {
            assert!(
                stop_regex_bound(pattern).is_err(),
                "{pattern} must be rejected"
            );
        }
    }

    /// Patterns Python compiles fine that this validator used to.
    #[test]
    fn ordinary_python_patterns_are_not_spuriously_rejected() {
        for pattern in [
            "(?-i:abc)", // scoped clearing group: legal anywhere
            "(?i-s:a)",  // mixed set/clear inside a scoped group
            r"\x41",
            r"a\x41b",
            "(?i)(?m)a", // several LEADING global flag groups
            "(?i)abc",
        ] {
            assert!(
                stop_regex_bound(pattern).is_ok(),
                "{pattern} is valid Python and must not be rejected"
            );
        }
        // The genuinely Rust-only forms still reject.
        for pattern in [r"\x{41}", "a(?i)b", "(?R)a"] {
            assert!(
                stop_regex_bound(pattern).is_err(),
                "{pattern} must be rejected"
            );
        }
    }

    /// A repetition count Python cannot honour: `u32::MAX` is its `MAXREPEAT` sentinel (`OverflowError`).
    #[test]
    fn oversized_repeat_counts_are_rejected() {
        for pattern in [
            "a{4294967295}",
            "a{4294967294}",
            "(?:a*){4294967294}",
            "a{1048576}",
            "a{0,4294967295}",
            "a{1048576,}",
        ] {
            assert!(
                stop_regex_bound(pattern).is_err(),
                "{pattern} must be rejected"
            );
        }
        // An ordinary count still works, and still yields a finite bound.
        assert_eq!(stop_regex_bound("a{200}").unwrap(), 200);
    }

    /// `\b{start}` is one zero-width assertion to Rust (bound) but `\b` plus the literal `{start}` to Python.
    #[test]
    fn b_brace_assertion_is_rejected_not_under_estimated() {
        assert!(stop_regex_bound(r"\b{start}xyz").is_err());
        assert!(stop_regex_bound(r"\b{end}").is_err());
        assert_eq!(
            stop_regex_bound(r"\bword").unwrap(),
            4,
            "plain \\b still works"
        );
    }

    /// Round 's under-estimate: `regex-syntax` reads `\<`/`\>` as GNU word-boundary assertions (width), CPython.
    #[test]
    fn gnu_word_boundary_escapes_are_rejected() {
        for pattern in [r"\<END\>", r"\<word", r"end\>"] {
            assert!(
                stop_regex_bound(pattern).is_err(),
                "{pattern} must be rejected"
            );
        }
        // A plain `<` is a literal in both and still bounds correctly.
        assert_eq!(stop_regex_bound("<END>").unwrap(), 5);
    }

    /// Repetition cost compounds down the nesting, so a per-node cap misses `(?:(?:a*){65535}){65535}` — bytes, compiles fine in Python.
    #[test]
    fn compounding_repetition_cost_is_rejected() {
        for pattern in [
            "(?:(?:a*){65535}){65535}",
            "(?:){1048575}x",
            "(?:a{100}){100}",
            "(?:a+)+b",
            "(a*)*b",
        ] {
            assert!(
                stop_regex_bound(pattern).is_err(),
                "{pattern} must be rejected"
            );
        }
        // Ordinary nesting still works.
        assert_eq!(stop_regex_bound("(?:ab){3}").unwrap(), 6);
        assert_eq!(stop_regex_bound(r"\d{6}").unwrap(), 6);
    }

    /// Deep nesting is rejected here rather than blowing Python's parser stack.
    #[test]
    fn deep_nesting_is_rejected_below_pythons_limit() {
        let nest = |n: usize| format!("{}a{}", "(".repeat(n), ")".repeat(n));
        assert!(
            stop_regex_bound(&nest(10)).is_ok(),
            "ordinary nesting is fine"
        );
        assert!(
            stop_regex_bound(&nest(400)).is_err(),
            "must be rejected here — Python raises RecursionError, not re.error"
        );
        assert!(stop_regex_bound(&nest(2000)).is_err());
    }

    /// Bounded patterns get their real length.
    #[test]
    fn stop_regex_bound_is_finite_when_bounded() {
        let len = |p: &str| stop_regex_bound(p).expect("valid pattern");
        assert_eq!(len(r"\d{6}"), 6);
        assert_eq!(len("abc"), 3);
        assert_eq!(len(r"^abc$"), 3); // anchors are zero-width
        assert_eq!(len("a|bbb"), 3); // alternation → max branch
        assert_eq!(len(r"(ab){3}"), 6);
        assert_eq!(len(r"a\d{2,5}"), 6);
        assert_eq!(len(r"\d+"), STOP_REGEX_MAX_LEN);
        assert_eq!(len(".*"), STOP_REGEX_MAX_LEN);
        assert_eq!(len(r"a{3,}"), STOP_REGEX_MAX_LEN);
    }
}
