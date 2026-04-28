[comment]: <> (# MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors)

<p align="center">
  <h1 align="center">MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors</h1>
  <p align="center">
    <a href="https://rmurai.co.uk/"><strong>Riku Murai*</strong></a>
    ·
    <a href="https://edexheim.github.io/"><strong>Eric Dexheimer*</strong></a>
    ·
    <a href="https://www.doc.ic.ac.uk/~ajd/"><strong>Andrew J. Davison</strong></a>
  </p>
  <p align="center">(* Equal Contribution)</p>

[comment]: <> (  <h2 align="center">PAPER</h2>)
  <h3 align="center"><a href="https://arxiv.org/abs/2412.12392">Paper</a> | <a href="https://youtu.be/wozt71NBFTQ">Video</a> | <a href="https://edexheim.github.io/mast3r-slam/">Project Page</a></h3>
  <div align="center"></div>

<p align="center">
    <img src="./media/teaser.gif" alt="teaser" width="100%">
</p>
<br>

> **About this fork.** This fork ships a Docker + [uv](https://docs.astral.sh/uv/) workflow on top of the upstream codebase so the project builds reproducibly on any modern Linux box without conda. A few targeted patches in [thirdparty/.../curope/](thirdparty/mast3r/dust3r/croco/models/curope/) were also needed to fix bit-rot against current PyTorch (2.5.1) and current CUDA toolchains. See [Changes from upstream](#changes-from-upstream) for the diff and rationale.

# Hardware requirements

The CUDA stack and the install scripts run on any NVIDIA GPU with the [container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) available. **The model itself needs ~7 GB of VRAM in the main process plus another ~2.5 GB in the backend process for loop closure / global BA**, so practical use needs a card with **12 GB or more** (RTX 3060 12 GB, RTX 4070 desktop, RTX 4080/4090, RTX 5090, A4000+, …). 8 GB cards (e.g. 4070 Laptop) will OOM with the default config.

# Getting Started

## Installation (Docker, recommended)

The Docker workflow is the path that "just works": it pins Ubuntu 22.04 + CUDA 12.4 toolkit + gcc 11 inside the image, so the host distro / compiler / system CUDA versions don't matter. Tested with a current Arch host.

Prerequisites on the host:

- Docker 25+ with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed, so containers can see the GPU. Quick check: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`.
- Recent NVIDIA driver (≥ 550, the runtime that ships in CUDA 12.4 wheels).

```bash
git clone <your-fork-url> --recursive
cd MASt3R-SLAM/

# If you cloned without --recursive:
# git submodule update --init --recursive

# Build the image (~10 min, one-time).
docker compose -f docker/compose.yml build

# Drop into a shell. The first run lazily executes `uv sync`, which
# downloads cu124 PyTorch wheels and compiles the three CUDA extensions
# (mast3r_slam_backends, lietorch, curope). That takes ~15 min and is
# cached afterwards in a named volume.
docker compose -f docker/compose.yml run --rm mast3r-slam
```

Inside the container:

- The repo is bind-mounted at `/app`, so edits on the host are live.
- The venv lives at `/opt/venv` (named volume `docker_venv`).
- All commands go through `uv run`, e.g. `uv run python main.py ...`.

To run a single command without entering a shell:

```bash
docker compose -f docker/compose.yml run --rm mast3r-slam \
    uv run python main.py --dataset datasets/tum/rgbd_dataset_freiburg1_room/ \
                          --config config/calib.yaml --no-viz
```

For the imgui/moderngl viewer (X11 forwarding), datasets mounted at `/data`, etc., uncomment the relevant blocks in [`docker/compose.yml`](docker/compose.yml).

If you change CUDA / Python / system deps in [`docker/Dockerfile`](docker/Dockerfile), rerun `docker compose -f docker/compose.yml build`. If you change the Python deps in `pyproject.toml`, just rerun `uv sync` inside the container.

## Installation — host uv (advanced)

<details>
<summary>Native install with uv. Works only on systems with CUDA 12.x + gcc ≤ 13.</summary>

The repository ships a `pyproject.toml` configured for [uv](https://docs.astral.sh/uv/) with the cu124 PyTorch index pinned, all upstream and local packages declared as `[tool.uv.sources]` path/git deps, and `extra-build-dependencies` set so the lietorch git source builds in an isolated env. Requires uv ≥ 0.8 (we used 0.11).

PyTorch's `cpp_extension` enforces that the system `nvcc` major+minor matches the CUDA the wheel was built against, so the cu124 wheels mandate a **CUDA 12.4** toolkit. There are no PyTorch wheels for CUDA 13.x. CUDA 12.4 in turn rejects host compilers newer than gcc 13, and gcc 15's libstdc++ uses builtins (`__is_array`, `__is_pointer`, …) that nvcc 12.4 cannot parse — so `-allow-unsupported-compiler` is not enough.

If your distro already has a usable CUDA 12.4 + gcc ≤ 13 (e.g. Ubuntu 22.04, Debian 12, older Fedora), `uv sync` should just work. Otherwise:

```bash
# Install CUDA 12.4 user-locally to ~/.local/cuda-12.4 (no sudo needed for
# the install itself; the runfile may want sudo for some optional bits).
bash scripts/install-cuda-12.4.sh

# Sibling gcc 13 — required for nvcc 12.4 to accept the headers.
sudo pacman -S gcc13         # Arch (currently only via AUR)
sudo dnf install gcc-13      # recent Fedora
# (Ubuntu 22.04 already ships gcc 11 by default, no extra step needed.)

# Per-shell exports before `uv sync`:
export CUDA_HOME=$HOME/.local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export CC=/usr/bin/gcc-13
export CXX=/usr/bin/g++-13
export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-13"

uv sync                    # base install
uv sync --extra mp4        # optional: torchcodec for faster mp4 decoding
```

Activate the venv with `source .venv/bin/activate` or prefix commands with `uv run`.

The `.cuda-12.4` toolkit and the `gcc13` package are removable with `rm -rf ~/.local/cuda-12.4` and `pacman -R gcc13` respectively.

</details>

## Installation — original conda

<details>
<summary>The original upstream conda instructions, kept verbatim.</summary>

```
conda create -n mast3r-slam python=3.11
conda activate mast3r-slam
```
Check the system's CUDA version with nvcc
```
nvcc --version
```
Install pytorch with **matching** CUDA version following:
```
# CUDA 11.8
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 12.1
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
# CUDA 12.4
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

Clone the repo and install the dependencies.
```
git clone https://github.com/rmurai0610/MASt3R-SLAM.git --recursive
cd MASt3R-SLAM/

# if you've clone the repo without --recursive run
# git submodule update --init --recursive

pip install -e thirdparty/mast3r
pip install -e thirdparty/in3d
pip install --no-build-isolation -e .


# Optionally install torchcodec for faster mp4 loading
pip install torchcodec==0.1
```

</details>

## Checkpoints

Setup the checkpoints for MASt3R and retrieval. The license for the checkpoints and more information on the datasets used is [here](https://github.com/naver/mast3r/blob/mast3r_sfm/CHECKPOINTS_NOTICE).

```bash
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P checkpoints/
```

## WSL Users

We have primarily tested on Ubuntu. If you are using WSL, please checkout to the windows branch and follow the above installation.
```
git checkout windows
```
This disables multiprocessing which causes an issue with shared memory as discussed [here](https://github.com/rmurai0610/MASt3R-SLAM/issues/21).

# Examples

Examples below use the Docker workflow. For host-uv, replace `docker compose -f docker/compose.yml run --rm mast3r-slam` with nothing and run from inside the activated venv.

```bash
docker compose -f docker/compose.yml run --rm mast3r-slam bash -c '
    bash ./scripts/download_tum.sh && \
    uv run python main.py \
        --dataset datasets/tum/rgbd_dataset_freiburg1_room/ \
        --config config/calib.yaml \
        --no-viz
'
```

Outputs land in `logs/<save_as>/` (default `logs/default/`):

- `<seq>.txt` — estimated trajectory in TUM format (`timestamp tx ty tz qx qy qz qw`)
- `<seq>.ply` — dense reconstruction
- `keyframes/<seq>/` — per-keyframe data

## Live Demo

Connect a realsense camera to the PC and run (host-uv only — passing a USB device to the container needs `--device=/dev/bus/usb` plumbing not enabled by default):

```
python main.py --dataset realsense --config config/base.yaml
```

## Running on a video

The system can process MP4 videos or folders of RGB images:

```bash
docker compose -f docker/compose.yml run --rm mast3r-slam \
    uv run python main.py --dataset <path/to/video>.mp4 --config config/base.yaml
docker compose -f docker/compose.yml run --rm mast3r-slam \
    uv run python main.py --dataset <path/to/folder> --config config/base.yaml
```

If the calibration parameters are known, supply `config/intrinsics.yaml`:

```bash
docker compose -f docker/compose.yml run --rm mast3r-slam \
    uv run python main.py --dataset <...> --config config/calib.yaml --calib config/intrinsics.yaml
```

# Datasets

```bash
bash ./scripts/download_tum.sh         # TUM-RGBD
bash ./scripts/download_7_scenes.sh    # 7-Scenes
bash ./scripts/download_euroc.sh       # EuRoC
bash ./scripts/download_eth3d.sh       # ETH3D SLAM
```

# Evaluations

All evaluation scripts run in single-threaded headless mode.

```bash
bash ./scripts/eval_tum.sh
bash ./scripts/eval_tum.sh --no-calib

bash ./scripts/eval_7_scenes.sh
bash ./scripts/eval_7_scenes.sh --no-calib

bash ./scripts/eval_euroc.sh
bash ./scripts/eval_euroc.sh --no-calib

bash ./scripts/eval_eth3d.sh
```

Trajectory accuracy against TUM-format ground truth (uses `evo`, already in the venv):

```bash
docker compose -f docker/compose.yml run --rm mast3r-slam \
    uv run evo_ape tum \
        datasets/tum/rgbd_dataset_freiburg1_room/groundtruth.txt \
        logs/default/rgbd_dataset_freiburg1_room.txt \
        --align --correct_scale -as
```

## Reproducibility

There may be minor differences between the released version and the paper results, since the multi-process refactor lands here. The paper results were measured on an RTX 4090; performance can drift slightly on different GPUs.

# Changes from upstream

This fork only changes what was needed to make the project build and run on a current Linux box with cu124 PyTorch wheels. The upstream behaviour is otherwise untouched. Summary:

| File | Change | Why |
|---|---|---|
| `pyproject.toml` | Full uv project: cu124 PyTorch index, path/git sources, `extra-build-dependencies` for lietorch, Python 3.11 pin. | Make `uv sync` resolve cleanly with PyTorch matching the bundled CUDA. |
| `setup.py` | Drop `sm_60`/`sm_61` from gencode, add `sm_89`. | sm_60/61 fail to build on CUDA 13 toolchains; sm_89 (Ada) avoids PTX JIT on RTX 40-series. |
| `thirdparty/mast3r/dust3r/croco/models/curope/pyproject.toml` *(new)* | Declare `[build-system]` with `setuptools` + `torch==2.5.1`. | The original setup.py had no pyproject.toml, so uv's isolated build env had no setuptools and no torch — and a floating `torch` would pull a newer version, producing an ABI-incompatible `.so`. |
| `thirdparty/mast3r/dust3r/croco/models/curope/setup.py` | Replace auto-detected gencode list with explicit `sm_70..sm_90`; add `-D_GLIBCXX_USE_CXX11_ABI=0`. | `torch.cuda.get_gencode_flags()` includes sm_100 (Blackwell), which nvcc 12.4 doesn't know about. The ABI flag matches libtorch's old C++11 ABI on Linux wheels — without it, the resulting `.so` references `std::__cxx11::basic_string` symbols that don't exist in libtorch. |
| `thirdparty/mast3r/dust3r/croco/models/curope/kernels.cu` | `tokens.type()` → `tokens.scalar_type()`. | The deprecated `Tensor::type()` API was removed in PyTorch 2.5; the new API returns `c10::ScalarType` directly, which is what `AT_DISPATCH_*` expects. |
| `docker/Dockerfile`, `docker/compose.yml`, `docker/entrypoint.sh` *(new)* | Reproducible Ubuntu 22.04 + CUDA 12.4 + gcc 11 + Python 3.11 + uv image; lazy `uv sync` on first container start; persistent venv volume. `shm_size: 8gb` to avoid `Bus error` from PyTorch shared-memory tensors. | Docker is the only path that's reliable across hosts today, given the gcc-15 / nvcc-12.4 incompatibility. |
| `scripts/install-cuda-12.4.sh` *(new)* | Helper that `wget`s the NVIDIA runfile and installs to `~/.local/cuda-12.4`. | Used only by the host-uv path. |
| `uv.lock` *(new)* | Resolved dependency lockfile. | Pin transitive deps for reproducibility across machines. |

The `mast3r_slam/` source itself, all configs, all evaluation scripts, and the upstream `thirdparty/mast3r/`, `thirdparty/in3d/`, and submodules are unchanged outside the curope subdir.

# Acknowledgement

We sincerely thank the developers and contributors of the many open-source projects that this code is built upon.
- [MASt3R](https://github.com/naver/mast3r)
- [MASt3R-SfM](https://github.com/naver/mast3r/tree/mast3r_sfm)
- [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM)
- [ModernGL](https://github.com/moderngl/moderngl)

# Citation

If you found this code/work to be useful in your own research, please considering citing the following:

```bibtex
@inproceedings{murai2024_mast3rslam,
  title={{MASt3R-SLAM}: Real-Time Dense {SLAM} with {3D} Reconstruction Priors},
  author={Murai, Riku and Dexheimer, Eric and Davison, Andrew J.},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025},
}
```
