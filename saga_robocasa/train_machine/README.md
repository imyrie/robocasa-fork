# RTX 5090 training machine: preparation

The advisor-provided RTX 5090 machine is for **policy training only**. No benchmark audit episodes and no
demonstration collection run there: those stay on the verified Rutgers nodes rlab3, rlab4 and rlab7, where the
physics and IK are bit-exact. Policy training does not start until the 100-demonstration dataset has passed
validation **and** the training protocol has been approved.

## Steps

1. **Environment** (run on the machine):

   ```bash
   bash setup_train_env.sh --base /LARGE_VOLUME/saga
   ```

   This creates the following under that directory:
   - a clean clone of `origin/saga-spot-adapter`;
   - robosuite at the frozen commit `5ce6643f`;
   - a Python 3.11 venv with CUDA PyTorch (`torch==2.7.1`, cu128 wheels, needed for Blackwell sm_120), numpy 2.2.5 and scipy 1.15.3, the same versions as the collection nodes;
   - the `data/`, `outputs/` and `logs/` directories;
   - `env_report.txt`, with nvidia-smi, torch and CUDA versions, the device name and capability, a GPU matmul check, `pip freeze` and the code commits.

   `--with-sim` also installs mujoco 3.3.1 and editable robosuite and robocasa. That is only needed later for closed-loop evaluation, and the robocasa assets are not part of the clone.

2. **Pilot transfer** (from a Rutgers login node; train pilots only):

   ```bash
   bash transfer_dataset.sh --src RUN_ROOT --dest USER@HOST:/LARGE_VOLUME/saga/data/pilot --pilot 0
   ```

   The script checks the source against the sha256 in the pilot's validation JSON, transfers with rsync, and re-checks on the destination with `sha256sum -c`.

3. **Loader-only smoke test** (on the machine; no model and no training step):

   ```bash
   python saga_robocasa/train_machine/loader_smoke_test.py --pilot /LARGE_VOLUME/saga/data/pilot/episode_000/attempt_1/episode_000
   ```

4. **Full dataset**, after the aggregator reports the dataset complete:

   ```bash
   bash transfer_dataset.sh --src RUN_ROOT --dest USER@HOST:/LARGE_VOLUME/saga/data/NAME
   ```

   The script verifies `SHA256SUMS` and the index files at both ends.

## Split discipline

Training code loads demonstrations **only** through `policy.benchmark_dataset.train_prefixes(train_index.json)`. Before any array is read, it checks:
- the release, schema, expert commit and manifest stamps;
- that every episode is a train episode (000–049);
- every file's checksum.

Held-out reference demonstrations (050–099) can be opened only through `heldout_reference_prefixes(..., purpose)`. It refuses training, normalization statistics, augmentation fitting, checkpoint selection and hyperparameter selection.
