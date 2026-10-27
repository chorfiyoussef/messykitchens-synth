#!/usr/bin/env bash
# Full MessyKitchens-Synthetic recreation: 600 training scenes per level (paper) plus validation
# scenes built only from the held-out validation objects, then validation, contact sheets and
# merged MOD manifests. Resumable: rerun the same command after an interruption.
#
#   bash scripts/run_full.sh                       # out/full, all GPUs, seed 0
#   OUT=out/full GPUS=0,1 SEED=0 bash scripts/run_full.sh
#
# Environment knobs: N_SCENES (600), N_VAL (30), WORKERS_PER_GPU (3), DEVICE_TYPE (OPTIX on
# NVIDIA, empty = BlenderProc's choice), CHUNK (10), LEVELS ("easy medium hard"), MOD_PREPARE
# (run MOD prepare --generate-training-gt on this many scenes per level; source mod_env.sh first).
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${OUT:-out/full}"
SEED="${SEED:-0}"
N_SCENES="${N_SCENES:-600}"
N_VAL="${N_VAL:-30}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-3}"
CHUNK="${CHUNK:-10}"
LEVELS="${LEVELS:-easy medium hard}"
if [ -z "${GPUS+x}" ]; then
    if command -v nvidia-smi > /dev/null; then
        GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
        DEVICE_TYPE="${DEVICE_TYPE-OPTIX}"
    else
        GPUS=""
    fi
fi
PY=.venv/bin/python
mkdir -p "${OUT}"

for level in ${LEVELS}; do
    args=(--level "${level}" --n_scenes "${N_SCENES}" --n_val "${N_VAL}" --seed "${SEED}" --out "${OUT}/${level}"
          --workers_per_gpu "${WORKERS_PER_GPU}" --chunk "${CHUNK}")
    [ -n "${GPUS}" ] && args+=(--gpus "${GPUS}")
    [ -n "${DEVICE_TYPE:-}" ] && args+=(--device_type "${DEVICE_TYPE}")
    echo "=== ${level}: $(date)"
    "${PY}" generate.py "${args[@]}"
    vargs=(--out "${OUT}/${level}")
    [ -n "${MOD_PREPARE:-}" ] && vargs+=(--mod_prepare "${MOD_PREPARE}")
    "${PY}" validate.py "${vargs[@]}" || echo "validation reported failures for ${level} (see ${OUT}/${level}/validation.json)"
    "${PY}" -m mksynth.viz "${OUT}/${level}" --sheet "${OUT}/${level}/contact_sheet.png" --columns 6
done
"${PY}" -m mksynth.report "${OUT}"
echo "=== done: $(date)"
