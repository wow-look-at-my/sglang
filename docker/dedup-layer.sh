#!/bin/sh
# "dedup-layer --begin" starts a RUN; "dedup-layer <dir>..." ends it.
set -eu
marker=/tmp/.dedup-layer-begin
if [ "${1:-}" = --begin ]; then
	touch "$marker"
	exit 0
fi
[ -e "$marker" ] || { echo "dedup-layer: run 'dedup-layer --begin' at the start of this RUN" >&2; exit 1; }
list=$(mktemp)
find "$@" -xdev -type f -size +0 -cnewer "$marker" > "$list"
echo "dedup-layer: $(wc -l < "$list") new files under $*"
fclones group --stdin --hidden --no-ignore < "$list" | fclones link
rm -f "$list" "$marker"
