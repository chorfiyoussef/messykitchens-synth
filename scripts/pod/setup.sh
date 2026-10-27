#!/usr/bin/env bash
# Set up a Linux GPU machine (tested: RunPod runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04,
# 2x L40S) for scene generation and for MOD's prepare step.
#
#   bash scripts/pod/setup.sh generator   # system packages, generator venv, Blender 4.2.1
#   bash scripts/pod/setup.sh gaps        # GAPS mshalign at the commit MOD pins
#   bash scripts/pod/setup.sh sam3d       # SAM3D environment at the commit MOD pins, plus MOD itself
#   HF_TOKEN=... bash scripts/pod/setup.sh weights   # gated facebook/sam-3d-objects weights
#
# Everything lands under $WS (default /workspace) so it survives a pod stop; system packages
# are reinstalled by running the `generator` step again after a restart.
set -euo pipefail

WS="${WS:-/workspace}"
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
EXTERNAL="${WS}/external"
SAM3D_ROOT="${EXTERNAL}/sam-3d-objects"
SAM3D_COMMIT=afdf6a31522d038c44c68a0bb57aa68827380797
GAPS_COMMIT=f78eb23e4b44378af1472c17a65d74b2c4e63c66
CONDA_DIR="${WS}/miniforge"
SAM3D_ENV="${WS}/envs/sam3d"
mkdir -p "${EXTERNAL}"

system_packages() {
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq build-essential git rsync tmux wget curl file unzip \
        mesa-common-dev libglu1-mesa-dev libosmesa6-dev libxi-dev libgl1-mesa-dev libglew-dev libjpeg-dev libpng-dev \
        libxrender1 libxxf86vm1 libxfixes3 libxkbcommon0 libsm6 libice6 libegl1 libxext6 libx11-6 libusb-1.0-0 libgomp1 > /dev/null
}

generator() {
    system_packages
    python3.11 -m venv "${REPO}/.venv"
    "${REPO}/.venv/bin/pip" install -q --upgrade pip
    "${REPO}/.venv/bin/pip" install -q -r "${REPO}/requirements.txt"
    # BlenderProc downloads its pinned Blender (4.2.1) into .blender on first use.
    printf 'import blenderproc as bproc\nimport bpy\nbproc.init()\nprint("BLENDER", bpy.app.version_string)\n' > /tmp/hello_bproc.py
    "${REPO}/.venv/bin/blenderproc" run --blender-install-path "${REPO}/.blender" /tmp/hello_bproc.py | grep BLENDER
}

gaps() {
    system_packages
    [ -d "${EXTERNAL}/gaps" ] || git clone -q https://github.com/tomfunkhouser/gaps.git "${EXTERNAL}/gaps"
    git -C "${EXTERNAL}/gaps" checkout -q "${GAPS_COMMIT}"
    make -C "${EXTERNAL}/gaps/pkgs" -j"$(nproc)" > /dev/null
    make -C "${EXTERNAL}/gaps/apps/mshalign" -j"$(nproc)" > /dev/null
    test -x "${EXTERNAL}/gaps/bin/x86_64/mshalign"
    if ldd "${EXTERNAL}/gaps/bin/x86_64/mshalign" | grep -q 'not found'; then
        echo "mshalign has unresolved shared libraries" >&2
        exit 1
    fi
    echo "GAPS_MSHALIGN=${EXTERNAL}/gaps/bin/x86_64/mshalign"
}

sam3d() {
    if [ ! -x "${CONDA_DIR}/bin/mamba" ]; then
        wget -q -O /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
        bash /tmp/miniforge.sh -b -p "${CONDA_DIR}"
    fi
    [ -d "${SAM3D_ROOT}" ] || git clone -q https://github.com/facebookresearch/sam-3d-objects.git "${SAM3D_ROOT}"
    git -C "${SAM3D_ROOT}" checkout -q "${SAM3D_COMMIT}"
    if [ ! -x "${SAM3D_ENV}/bin/python" ]; then
        "${CONDA_DIR}/bin/mamba" env create -q -p "${SAM3D_ENV}" -f "${SAM3D_ROOT}/environments/default.yml"
    fi
    local py="${SAM3D_ENV}/bin/python"
    export PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10  # the kaolin wheel host (S3) is slow at times
    # MOD's instructions also list https://pypi.ngc.nvidia.com, which no longer resolves (Oct 2026);
    # the NVIDIA wheels it served are on PyPI.
    export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121"
    (cd "${SAM3D_ROOT}" && "${py}" -m pip install -q -e '.[dev]')
    (cd "${SAM3D_ROOT}" && "${py}" -m pip install -q -e '.[p3d]')
    export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
    (cd "${SAM3D_ROOT}" && "${py}" -m pip install -q -e '.[inference]')
    (cd "${SAM3D_ROOT}" && "${py}" patching/hydra)
    "${py}" -m pip install -q -e "${REPO}/third_party/MessyKitchens[train,iou]"
    "${py}" -m pip install -q 'huggingface-hub[cli]<1.0'
    "${py}" "${REPO}/third_party/MessyKitchens/scripts/setup_sam3d.py" --sam3d-root "${SAM3D_ROOT}"
}

weights() {
    : "${HF_TOKEN:?set HF_TOKEN (read access to the gated facebook/sam-3d-objects repo)}"
    HF_TOKEN="${HF_TOKEN}" "${SAM3D_ENV}/bin/hf" download facebook/sam-3d-objects --repo-type model \
        --local-dir "${SAM3D_ROOT}/checkpoints/hf-download" --max-workers 1 > /dev/null
    find "${SAM3D_ROOT}/checkpoints/hf-download" -name pipeline.yaml
}

env_file() {
    # Source this before running validate.py --mod_prepare.
    cat > "${WS}/mod_env.sh" <<EOF
export SAM3D_ROOT="${SAM3D_ROOT}"
export SAM3D_PIPELINE_CONFIG="\$(find "${SAM3D_ROOT}/checkpoints/hf-download" -name pipeline.yaml | head -1)"
export MOD_PYTHON="${SAM3D_ENV}/bin/python"
export SAM3D_PYTHON="${SAM3D_ENV}/bin/python"
export GAPS_MSHALIGN="${EXTERNAL}/gaps/bin/x86_64/mshalign"
export PATH="${SAM3D_ENV}/bin:\${PATH}"
EOF
    echo "wrote ${WS}/mod_env.sh"
}

for step in "$@"; do
    case "${step}" in
        generator|gaps|sam3d|weights|env_file) "${step}" ;;
        *) echo "unknown step ${step}" >&2; exit 2 ;;
    esac
done
