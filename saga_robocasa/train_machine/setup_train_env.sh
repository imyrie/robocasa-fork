#!/bin/bash
# Prepare the RTX 5090 policy-training machine (run ON that machine). Environment preparation only: no benchmark
# audit episodes, no demonstration collection and no training are run here.
#
#   bash setup_train_env.sh --base /path/on/large/volume [--repo-url URL] [--with-sim]
#
# Creates under BASE:
#   code/robocasa     clean clone of origin/saga-spot-adapter (checked: no local changes, HEAD = origin/saga-spot-adapter)
#   code/robosuite    robosuite at the commit the benchmark was frozen with (5ce6643f), detached, clean
#   envs/saga_train   Python 3.11 venv: CUDA PyTorch (cu128 wheels; Blackwell / sm_120 needs CUDA >= 12.8) + pinned
#                     numpy/scipy as on the collection nodes; --with-sim also installs mujoco 3.3.1 and editable
#                     robosuite/robocasa (only needed later for closed-loop evaluation)
#   data/ outputs/ logs/
#   env_report.txt    nvidia-smi, driver, torch/CUDA versions, GPU name and capability, pip freeze, git commits
set -euo pipefail
BASE="" REPO_URL="git@github.com:imyrie/robocasa-fork.git" SIM=0
ROBOSUITE_URL="https://github.com/ARISE-Initiative/robosuite.git" ROBOSUITE_COMMIT=5ce6643f3092639d08f7b0f90ed1c6a84f50552c
TORCH="torch==2.7.1" TORCH_INDEX="https://download.pytorch.org/whl/cu128"
while [ $# -gt 0 ]; do
  case "$1" in
    --base) BASE=$2; shift 2;;
    --repo-url) REPO_URL=$2; shift 2;;
    --with-sim) SIM=1; shift;;
    *) echo "unknown argument $1"; exit 2;;
  esac
done
[ -n "$BASE" ] || { echo "--base is required (a directory on the large storage volume)"; exit 2; }
mkdir -p "$BASE"/{code,envs,data,outputs,logs}
df -h "$BASE" | tail -1
cd "$BASE/code"
if [ ! -d robocasa ]; then git clone --branch saga-spot-adapter --single-branch "$REPO_URL" robocasa; fi
git -C robocasa fetch origin saga-spot-adapter
[ -z "$(git -C robocasa status --porcelain)" ] || { echo "code/robocasa has local changes"; exit 2; }
git -C robocasa checkout -q saga-spot-adapter && git -C robocasa merge -q --ff-only origin/saga-spot-adapter
[ "$(git -C robocasa rev-parse HEAD)" = "$(git -C robocasa rev-parse origin/saga-spot-adapter)" ] || { echo "not at origin/saga-spot-adapter"; exit 2; }
if [ ! -d robosuite ]; then git clone "$ROBOSUITE_URL" robosuite; fi
git -C robosuite fetch -q origin && git -C robosuite checkout -q "$ROBOSUITE_COMMIT"
[ "$(git -C robosuite rev-parse HEAD)" = "$ROBOSUITE_COMMIT" ] && [ -z "$(git -C robosuite status --porcelain)" ] || { echo "robosuite not clean at $ROBOSUITE_COMMIT"; exit 2; }
PY=$(command -v python3.11 || true)
[ -n "$PY" ] || { echo "python3.11 not found (install it, or create the env with conda: conda create -p $BASE/envs/saga_train python=3.11)"; exit 2; }
[ -d "$BASE/envs/saga_train" ] || "$PY" -m venv "$BASE/envs/saga_train"
source "$BASE/envs/saga_train/bin/activate"
pip install -q --upgrade pip
pip install -q "$TORCH" --index-url "$TORCH_INDEX"
pip install -q numpy==2.2.5 scipy==1.15.3 tqdm
if [ $SIM = 1 ]; then
  pip install -q mujoco==3.3.1
  pip install -q -e "$BASE/code/robosuite" && pip install -q -e "$BASE/code/robocasa"
fi
{
  echo "date $(date -Is) host $(hostname)"
  echo "robocasa $(git -C "$BASE/code/robocasa" rev-parse HEAD)  robosuite $(git -C "$BASE/code/robosuite" rev-parse HEAD)"
  nvidia-smi
  python - <<'PYEOF'
import torch, numpy, scipy
print("torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version())
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
    x = torch.randn(1024, 1024, device="cuda"); print("matmul ok", float((x @ x).sum().isfinite()))
print("numpy", numpy.__version__, "scipy", scipy.__version__)
PYEOF
  pip freeze
} | tee "$BASE/env_report.txt"
echo "environment ready: source $BASE/envs/saga_train/bin/activate"
