#!/bin/sh
set -eu
dir=$1
archs=$2
[ -n "$archs" ] || exit 0
list=$(mktemp)
find "$dir" -type f -printf '%s\t%p\n' | awk -F'\t' -v archs="$archs" -v out="$list" '
BEGIN {
	m = split(archs, t, ";")
	n = 0
	for (i = 1; i <= m; i++) {
		fam = t[i] ~ /f$/
		gsub(/[^0-9]/, "", t[i])
		if (t[i] == "") continue
		c = t[i] + 0
		# A family target ("100f") stands for every GPU of that major from its minor up.
		last = fam ? int(c / 10) * 10 + 9 : c
		for (; c <= last; c++) cc[++n] = c
	}
}
function runs(tag, sfx,    i, d) {
	d = tag + 0
	for (i = 1; i <= n; i++) {
		if (sfx == "a" && d == cc[i]) return 1
		if (sfx != "a" && int(d / 10) == int(cc[i] / 10) && cc[i] % 10 >= d % 10) return 1
	}
	return 0
}
{
	name = $2
	sub(/.*\//, "", name)
	if (!match(name, /[Ss][Mm]_?[0-9]+[af]?/)) next
	tag = substr(name, RSTART, RLENGTH)
	sfx = tag ~ /[af]$/ ? substr(tag, length(tag)) : ""
	gsub(/[^0-9]/, "", tag)
	if (runs(tag, sfx)) next
	print $2 > out
	files++
	bytes += $1
}
END { printf "prune-cubins: removing %d files (%d MB) that no arch in GPU_ARCHS=%s can run\n", files, bytes / 1048576, archs }'
xargs -r -d '\n' rm -f < "$list"
rm -f "$list"
