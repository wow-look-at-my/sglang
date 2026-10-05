#!/bin/bash
# prune-cuda-libs.sh GPU_ARCHS Rebuilds the CUDA math libraries from their static archives with SASS for GPU_ARCHS only.
set -euo pipefail
archs=$1
[ -n "$archs" ] || exit 0
command -v g++ >/dev/null || { echo "prune-cuda-libs: g++ is missing" >&2; exit 1; }
lib=/usr/local/cuda/lib64
manifest=/usr/local/share/cuda-unpruned.sha256
work=$(mktemp -d)

# "120a" keeps sm_120 and sm_120a; a family target "100f" keeps every minor of its major from its own minor up.
gencode=()
IFS=';' read -ra entries <<< "$archs"
for e in "${entries[@]}"; do
	cc=${e//[!0-9]/}
	[ -n "$cc" ] || continue
	last=$cc
	[[ $e == *f ]] && last=$(( cc / 10 * 10 + 9 ))
	for (( c = cc; c <= last; c++ )); do
		for s in "" a f; do
			gencode+=(-gencode "arch=compute_${c}${s},code=sm_${c}${s}")
		done
	done
done

exports() {
	readelf --dyn-syms -W "$1" | awk '$7 != "UND" && $5 == "GLOBAL" && $8 ~ /@@/ { sub(/@@.*/, "", $8); print $8 }' | sort -u
}

# rebuild SONAME "static archives" [dynamic deps...]
rebuild() {
	local so=$1 archives=$2
	shift 2
	local orig pruned=() a
	orig=$(readlink -f "$lib/$so")
	for a in $archives; do
		nvprune "${gencode[@]}" -o "$work/$a" "$lib/$a"
		pruned+=("$work/$a")
	done
	exports "$orig" > "$work/$so.syms"
	{ echo "$so {"; echo " global:"; sed 's/$/;/' "$work/$so.syms"; echo " local: *;"; echo "};"; } > "$work/$so.map"
	g++ -shared -s -static-libstdc++ -Wl,-z,defs -o "$work/$so" -Wl,-soname,"$so" -Wl,--version-script="$work/$so.map" \
		-Wl,--whole-archive "${pruned[@]}" -Wl,--no-whole-archive \
		"$lib/libculibos.a" "$lib/libcudart_static.a" -L "$lib" "$@" -lpthread -ldl -lrt -lm
	if ! cmp -s "$work/$so.syms" <(exports "$work/$so"); then
		echo "prune-cuda-libs: $so does not export the same symbols as $orig" >&2
		exit 1
	fi
	echo "$(sha256sum "$orig" | cut -d' ' -f1)  $so" >> "$manifest"
	echo "prune-cuda-libs: $so $(( $(stat -c %s "$orig") / 1048576 )) MB -> $(( $(stat -c %s "$work/$so") / 1048576 )) MB"
	cp "$work/$so" "$orig"
	rm -f "${pruned[@]}"
}

rebuild libcublasLt.so.13 libcublasLt_static.a
rebuild libcublas.so.13 libcublas_static.a -l:libcublasLt.so.13
rebuild libcurand.so.10 libcurand_static.a
rebuild libcusparse.so.12 libcusparse_static.a -l:libnvJitLink.so.13
rebuild libcusolver.so.12 "libcusolver_static.a libcusolver_lapack_static.a libcusolver_metis_static.a" \
	-l:libcusparse.so.12 -l:libcublas.so.13 -l:libcublasLt.so.13 -l:libnvJitLink.so.13
rm -rf "$work"
