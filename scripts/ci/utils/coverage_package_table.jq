# Input: `coverage json` output. Output: markdown total line plus a per-package table.
# A package is sglang/<dir> or, under the large srt tree, sglang/srt/<dir>.
def pkg:
	sub("^.*?python/"; "") | split("/") | .[0:-1]
	| if length >= 2 and .[1] == "srt" then .[0:3] else .[0:2] end
	| join("/");

def pct($covered; $stmts):
	if $stmts == 0 then "-" else "\((($covered * 10000 / $stmts) | round) / 100)%" end;

.totals as $t
| [.files | to_entries[] | {pkg: (.key | pkg), covered: .value.summary.covered_lines, stmts: .value.summary.num_statements}]
| group_by(.pkg)
| map({pkg: .[0].pkg, covered: (map(.covered) | add), stmts: (map(.stmts) | add)})
| sort_by(-.stmts)
| "### Engine line coverage (python/sglang): \(pct($t.covered_lines; $t.num_statements)), \($t.covered_lines) / \($t.num_statements) statements",
	"",
	"| package | statements | covered | line % |",
	"|---|---:|---:|---:|",
	(.[] | "| \(.pkg) | \(.stmts) | \(.covered) | \(pct(.covered; .stmts)) |")
