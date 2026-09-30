# Statement-weighted coverage per file from a Go cover profile written with
# -coverprofile and -coverpkg=./..., so every package's statements count
# whichever test binary ran them.
#
#   awk -f scripts/filecov.awk cover.out          per-file table
#   awk -v min=95 -f scripts/filecov.awk cover.out fails (exit 1) when a file
#                                                  named in FILES is under min%
/^mode:/ { next }
{
  split($1, loc, ":"); f = loc[1]
  n = $2 + 0
  if (!($1 in seen)) { seen[$1] = 1; stmts[f] += n }
  if (($3 + 0) > 0 && !($1 in hit)) { hit[$1] = 1; cov[f] += n }
}
END {
  bad = 0
  split(FILES, want, " ")
  for (f in stmts) {
    pct = 100 * cov[f] / stmts[f]
    mark = ""
    for (i in want) if (index(f, want[i]) > 0 && min != "" && pct < min + 0) { mark = "  UNDER " min "%"; bad = 1 }
    printf "%6.1f%% %5d/%-5d %s%s\n", pct, cov[f], stmts[f], f, mark
  }
  if (bad) exit 1
}
