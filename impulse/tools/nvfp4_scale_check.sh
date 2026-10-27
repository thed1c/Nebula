#!/bin/bash
# nvfp4_scale_check.sh: the GPU checks for NVFP4's second-level ".scale" tensors and the 512-bit NVFP4 CPU rows.
# Runs on one machine with one card; nothing here is installed or kept running.
#
#   B=<build>/bin D=<dir with the files> CARD=<CUDA_VISIBLE_DEVICES value> bash nvfp4_scale_check.sh ka|kl|place|all
#
# Files in D (tools/nvfp4_scale_variant.py makes the *-scaled ones from llama-quantize's NVFP4 files):
#   q36-nvfp4.gguf, q36-nvfp4-scaled.gguf   Qwen3.6-35B-A3B, experts / attention / delta-net / shared expert / output NVFP4
#   mtp-nvfp4.gguf, mtp-nvfp4-scaled.gguf   its MTP drafter, the same
#   ka.json                                  the known-answer prompt (a chat request, ~0.6K tokens)
#   kl.tokens                                the KL text: 32 prompt + 3914 forced tokens (route-oracle ROUTE_TOKENIZE_ONLY)
# B: impulse-qwen35, route-oracle (llama.cpp b6b003d's own library), logit-cmp, pool-scales-test (IMPULSE_TOOLS=ON).
#
# ka (known answer, ~15 min): the scaled variant computes exactly the original's function (variant x scale == original,
#    value for value), so with scales applied every run on the variant must give the original's logits, and Impulse must
#    give llama.cpp's on the same file. 128 greedy tokens after a ~600-token prompt (two prompt ubatches, GPU-offloaded
#    experts; then one-token steps):
#    KA1 llama.cpp: variant == original (the known answer holds in llama.cpp itself)
#    KA2 Impulse plain --n-cpu-moe 20 (ggml's expert paths, half the layers' experts in RAM) == llama.cpp, variant
#    KA3 Impulse --pool (the CPU pool's NVFP4 rows, exact; per-job scales) == llama.cpp, variant
#    KA4 Impulse plain == llama.cpp, original (the baseline)
#    KA5 Impulse slot cache + adapt + async swaps + overlap + dcheck: variant == original, twin check clean
#    KA6 Impulse lent reads through the expert ring (IMPULSE_RING=2, --prefill-ubatch auto --lend): variant == original
#    KA7 Impulse MTP drafter (mtp-force, 3 drafts): variant pair == original pair (logits and accepted drafts)
#    KA8 pool-scales-test (CPU): the pool's per-job scales bit for bit, every format
# kl (the 512-bit NVFP4 CPU rows' quality gate, ~15 min): every expert on the CPU (--n-cpu-moe 40, --pool), the original
#    NVFP4 file, teacher-forced over the KL text: exact rows (IMPULSE_NVFP4_512=0) vs 512-bit rows (=1; on an AVX-512 CPU
#    without VNNI the BW twin, bit for bit the VNNI rows). Pass: mean KL <= 0.005 (max KL and top-1 agreement reported).
# place (~5 min): one run with the pool's per-socket split and the experts placed per socket (IMPULSE_NUMA_NODES as set
#    by the caller), both sockets' CPUs: the placement and numa_maps lines, and the logits against KA3's.
set -u
B=${B:?build bin dir}; D=${D:?data dir}; CARD=${CARD:?CUDA_VISIBLE_DEVICES}
cd "$D" || exit 1
export CUDA_VISIBLE_DEVICES=$CARD
OUT=$D/out; mkdir -p "$OUT"
SUM=$OUT/summary.txt
row() { printf '%-4s %-62s %-26s %s\n' "$1" "$2" "$3" "$4" | tee -a "$SUM"; }
TAG=${TAG:-$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null | head -1)}
CPU=$(lscpu | sed -n 's/^Model name: *//p' | head -1)
T=${THREADS:-16}
NUMA1=${NUMA1-numactl --cpunodebind=0 --membind=0}   # one socket, unless the caller says otherwise
V=248320   # Qwen3.6's vocabulary (logit-cmp)
js() { python3 -c "import json,sys; d=json.load(open('$1')); print($2)" 2>/dev/null; }
same() { cmp -s "$1" "$2"; }

q35() {   # name model [args...] -> $OUT/name.{json,log,hash,logits}
  local n=$1 m=$2; shift 2
  Q35_LOGIT_HASH=$OUT/$n.hash Q35_LOGITS_OUT=${LOGITS_OUT:-/dev/null} timeout 3600 $NUMA1 $B/impulse-qwen35 -m "$m" -c 8192 -t $T "$@" \
    --out $OUT/$n > $OUT/$n.log 2>&1
  tail -1 $OUT/$n.log > $OUT/$n.json
}
ident() {   # the json of a run scored against an oracle: identical, or how far off
  local j=$1
  local s=$(js $j 'd["steps_compared"]') i=$(js $j 'd["logits_bit_identical"]') m=$(js $j 'd["max_abs_diff"]') k=$(js $j 'd["kl_mean"]')
  [ -n "$s" ] && [ "$s" -gt 0 ] && [ "$s" = "$i" ] && echo "PASS bit-identical ($s steps)" || echo "FAIL $i/$s identical, max|d| $m, KL $k"
}

ka() {
  export IMPULSE_NVFP4_512=0   # the exact NVFP4 CPU rows: bit for bit with llama.cpp's CPU arithmetic (512-bit: the kl gate)
  local ORA="-ngl 99 --n-cpu-moe 20 -fa on -c 8192 -ctk f16 -ctv f16 -b 2048 -ub 512 -t $T --temp 0"
  # KA1: llama.cpp greedy on the original, then forced on the variant
  ROUTE_LOGITS=1 ROUTE_REQ=ka.json ROUTE_OUT=$OUT/ka.orig ROUTE_NGEN=128 timeout 1800 $NUMA1 $B/route-oracle -m q36-nvfp4.gguf $ORA > $OUT/ka.orig.log 2>&1
  ROUTE_LOGITS=1 ROUTE_FORCE=$OUT/ka.orig.tokens ROUTE_OUT=$OUT/ka.var timeout 1800 $NUMA1 $B/route-oracle -m q36-nvfp4-scaled.gguf $ORA > $OUT/ka.var.log 2>&1
  if [ ! -s $OUT/ka.orig.logits ] || [ ! -s $OUT/ka.var.logits ]; then row KA1 "llama.cpp did not run (see out/ka.*.log)" "$TAG" FAIL; return; fi
  same $OUT/ka.orig.logits $OUT/ka.var.logits && r="PASS bit-identical" || r="DIFF $($B/logit-cmp $OUT/ka.orig.logits $OUT/ka.var.logits $V)"
  row KA1 "llama.cpp b6b003d: scaled variant == original" "$TAG, llama.cpp CUDA" "$r"
  local TK="--tokens $OUT/ka.orig.tokens --mode force --n-cpu-moe 20"
  q35 ka2 q36-nvfp4-scaled.gguf $TK --oracle $OUT/ka.var.logits
  row KA2 "Impulse ggml expert paths (n-cpu-moe 20) == llama.cpp, variant" "$TAG + CPU" "$(ident $OUT/ka2.json)"
  q35 ka3 q36-nvfp4-scaled.gguf $TK --oracle $OUT/ka.var.logits --pool --pool-threads $T
  row KA3 "Impulse CPU pool NVFP4 rows (exact) == llama.cpp, variant" "$TAG + $CPU" "$(ident $OUT/ka3.json)"
  q35 ka4 q36-nvfp4.gguf $TK --oracle $OUT/ka.orig.logits
  row KA4 "Impulse ggml expert paths == llama.cpp, original (baseline)" "$TAG + CPU" "$(ident $OUT/ka4.json)"
  local SL="--pool --pool-threads $T --cache-slots 16 --adapt --adapt-budget 2 --async-swaps --overlap --dcheck-layer 3"
  q35 ka5v q36-nvfp4-scaled.gguf $TK $SL; q35 ka5o q36-nvfp4.gguf $TK $SL
  local bad=$(js $OUT/ka5v.json 'd["dcheck_bad"]') rows=$(js $OUT/ka5v.json 'd["dcheck_rows"]')
  same $OUT/ka5v.hash $OUT/ka5o.hash && [ -s $OUT/ka5v.hash ] && [ "$bad" = 0 ] && r="PASS same logits, twin check $rows rows clean" || r="FAIL (hash same: $(same $OUT/ka5v.hash $OUT/ka5o.hash && echo yes || echo no), twin $bad/$rows bad)"
  row KA5 "Impulse slot cache hits + pool misses + overlap: variant == orig" "$TAG + $CPU" "$r"
  local RD="--pool --pool-threads $T --cache-slots 16 --adapt --async-swaps --prefill-ubatch auto --lend"
  IMPULSE_RING=2 q35 ka6v q36-nvfp4-scaled.gguf $TK $RD; IMPULSE_RING=2 q35 ka6o q36-nvfp4.gguf $TK $RD
  same $OUT/ka6v.hash $OUT/ka6o.hash && [ -s $OUT/ka6v.hash ] && r=PASS || r=FAIL
  row KA6 "Impulse lent read through the expert ring: variant == original" "$TAG + $CPU" "$r same logits"
  local MT="--tokens $OUT/ka.orig.tokens --mode mtp-force --n-cpu-moe 20 --n-draft 3"
  q35 ka7v q36-nvfp4-scaled.gguf $MT --mtp mtp-nvfp4-scaled.gguf; q35 ka7o q36-nvfp4.gguf $MT --mtp mtp-nvfp4.gguf
  local av=$(js $OUT/ka7v.json '(d["drafted"], d["accepted"])') ao=$(js $OUT/ka7o.json '(d["drafted"], d["accepted"])')
  same $OUT/ka7v.hash $OUT/ka7o.hash && [ -s $OUT/ka7v.hash ] && [ "$av" = "$ao" ] && r="PASS same logits, drafts $av" || r="FAIL drafts $av vs $ao"
  row KA7 "Impulse MTP drafter (eh_proj, attn, experts, head scales)" "$TAG" "$r"
  [ -x $B/pool-scales-test ] && { $NUMA1 $B/pool-scales-test 8 > $OUT/ka8.log 2>&1 && r="PASS" || r="FAIL"; } || r="not built"
  row KA8 "pool-scales-test: per-job scales bit for bit, every format" "$CPU" "$r ($(grep -m1 'NVFP4 rows' $OUT/ka8.log 2>/dev/null))"
}

kl() {
  local A="--tokens kl.tokens --mode force --n-cpu-moe 40 --pool --pool-threads $T"
  LOGITS_OUT=$OUT/kl.exact.logits IMPULSE_NVFP4_512=0 IMPULSE_NVFP4_EXACT=1 q35 kl-exact q36-nvfp4.gguf $A
  LOGITS_OUT=$OUT/kl.512.logits IMPULSE_NVFP4_512=1 q35 kl-512 q36-nvfp4.gguf $A
  local r=$($B/logit-cmp $OUT/kl.exact.logits $OUT/kl.512.logits $V 1000000 $OUT/kl.steps)
  echo "$r" > $OUT/kl.json
  local m=$(js $OUT/kl.json 'd["kl_mean"]') x=$(js $OUT/kl.json 'd["kl_max"]') t=$(js $OUT/kl.json 'd["top1_rate"]') n=$(js $OUT/kl.json 'd["steps"]')
  local tps=$(js $OUT/kl-exact.json 'round(d["decode_tok_s"],1)')/$(js $OUT/kl-512.json 'round(d["decode_tok_s"],1)')
  python3 -c "import sys; sys.exit(0 if ${m:-1} <= 0.005 and ${n:-0} > 2000 else 1)" && g=PASS || g=FAIL
  row KL "512-bit NVFP4 rows vs exact, all experts on the CPU ($n steps)" "$CPU (+$TAG)" "$g mean KL $m, max $x, top-1 $t, tok/s $tps"
}

place() {
  export IMPULSE_NVFP4_512=0
  local A="--tokens $OUT/ka.orig.tokens --mode force --n-cpu-moe 40 --pool --pool-threads ${PLACE_THREADS:-34} --cache-slots 16 --adapt --async-swaps"
  NUMA1= q35 place q36-nvfp4-scaled.gguf $A --oracle $OUT/ka.var.logits
  grep -E "NUMA placement|numa_maps|strata cpu pool|IMPULSE_NUMA_NODES|second-level" $OUT/place.log | head -8 | tee -a "$SUM"
  grep -q "NUMA placement: slice 0" $OUT/place.log && r=PASS || r=FAIL
  row PL "two-socket split + per-socket placement (IMPULSE_NUMA_NODES)" "$CPU, both sockets" "$r placed; vs llama.cpp $(js $OUT/place.json '"top-1 %d/%d, KL %.2e" % (d["argmax_same"], d["steps_compared"], d["kl_mean"])')"
}

echo "== $(date) ${1:-all} on $TAG / $CPU" | tee -a "$SUM"
case "${1:-all}" in
  ka) ka;; kl) kl;; place) place;;
  all) ka; kl; place;;
  *) echo "ka|kl|place|all"; exit 2;;
esac
