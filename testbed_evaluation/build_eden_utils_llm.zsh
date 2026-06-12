#!/usr/bin/env zsh
unsetopt errexit nounset
source ~/.zshrc
setopt errexit nounset

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-llm}"
CUDA_HOME_DIR="${CUDA_HOME:-/share/apps/cuda-11.8}"
GCC_HOME="${GCC_HOME:-/share/apps/gcc-8.3}"
CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9}"
MAX_JOBS_VALUE="${MAX_JOBS:-4}"
PRINT_ONLY=0

usage() {
  cat <<'EOF'
Usage:
  ./build_eden_utils_llm.zsh [--print-only]

Environment overrides:
  CONDA_ENV              Conda environment to activate. Default: llm
  CUDA_HOME              CUDA toolkit root. Default: /share/apps/cuda-11.8
  GCC_HOME               GCC root. Default: /share/apps/gcc-8.3
  TORCH_CUDA_ARCH_LIST   CUDA arch list. Default: 8.0;8.6;8.9
  MAX_JOBS               Build parallelism. Default: 4
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --print-only)
      PRINT_ONLY=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

cat <<EOF
source ~/.zshrc
conda activate $CONDA_ENV
cd $REPO_DIR/cuda_kernels
CUDA_HOME=$CUDA_HOME_DIR \\
PATH=$GCC_HOME/bin:$CUDA_HOME_DIR/bin:\$PATH \\
LD_LIBRARY_PATH=$GCC_HOME/lib64:$CUDA_HOME_DIR/lib64:\${LD_LIBRARY_PATH:-} \\
CC=$GCC_HOME/bin/gcc \\
CXX=$GCC_HOME/bin/g++ \\
TORCH_CUDA_ARCH_LIST='$CUDA_ARCH_LIST' \\
MAX_JOBS=$MAX_JOBS_VALUE \\
python setup.py build_ext --inplace
EXT_SUFFIX=\$(python - <<'PY'
import sysconfig
print(sysconfig.get_config_var("EXT_SUFFIX"))
PY
)
cp "$REPO_DIR/cuda_kernels/src/srrcomp/eden_utils\${EXT_SUFFIX}" "$REPO_DIR/testbed_evaluation/new_comm_hooks/"
EOF

if (( PRINT_ONLY )); then
  exit 0
fi

conda activate "$CONDA_ENV"

(
  cd "$REPO_DIR/cuda_kernels"
  CUDA_HOME="$CUDA_HOME_DIR" \
  PATH="$GCC_HOME/bin:$CUDA_HOME_DIR/bin:$PATH" \
  LD_LIBRARY_PATH="$GCC_HOME/lib64:$CUDA_HOME_DIR/lib64:${LD_LIBRARY_PATH:-}" \
  CC="$GCC_HOME/bin/gcc" \
  CXX="$GCC_HOME/bin/g++" \
  TORCH_CUDA_ARCH_LIST="$CUDA_ARCH_LIST" \
  MAX_JOBS="$MAX_JOBS_VALUE" \
  python setup.py build_ext --inplace
)

EXT_SUFFIX="$(python - <<'PY'
import sysconfig
print(sysconfig.get_config_var("EXT_SUFFIX"))
PY
)"
cp "$REPO_DIR/cuda_kernels/src/srrcomp/eden_utils${EXT_SUFFIX}" "$REPO_DIR/testbed_evaluation/new_comm_hooks/"
echo "Copied eden_utils${EXT_SUFFIX} into testbed_evaluation/new_comm_hooks/"
