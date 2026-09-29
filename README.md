# SS-SD: Kinematic-Conditioned Stable Diffusion for Surgical Suturing

This project fine-tunes Stable Diffusion 1.5 to generate JIGSAWS
**Suturing** video frames directly from surgical robot kinematics and
gesture labels — no text prompts.

The CLIP text encoder is **replaced entirely** by a small
`KinematicEncoder` that projects a 76-dim kinematics vector plus an
integer gesture label into the U-Net's cross-attention space. The U-Net
is then adapted with LoRA on its cross-attention projections
(`to_q/to_k/to_v/to_out.0`). The VAE stays frozen.

```
kinematics (76-dim)  ─┐
                       ├─► KinematicEncoder ─► [B, 77, 768] ──► SD U-Net (LoRA) ──► latent ──► VAE.decode ──► RGB frame
gesture id (int)     ─┘                                          ▲
                                                                 │
                                                         noisy latent + timestep
```

**Terminology.** *Diffusion models* are a broad family of generative models.
**Stable Diffusion** is a *latent* diffusion system: a frozen **VAE** maps
images to a compact latent space; a **U-Net** predicts noise (or the
velocity) in that space; a **scheduler** (e.g. DDPM / DDIM) defines the
corruption process; and **conditioning** (originally CLIP text embeddings)
feeds **cross-attention** inside the U-Net. This project keeps that SD
backbone but replaces text conditioning with **kinematic-guided**
cross-attention — i.e. a **kinematic-guided latent diffusion** model.

**Inference** mirrors standard latent diffusion sampling: Gaussian noise in
VAE latent space → iterative denoising with the U-Net conditioned on
embeddings from `KinematicEncoder` → VAE decode to RGB.

### JIGSAWS kinematic layout (76 columns per frame)

Each row of the kinematics file matches one video frame (30 Hz). Values are
documented in code as [`jigsaws_kinematics_layout.py`](src/suturing_pipeline/data/jigsaws_kinematics_layout.py).
Layout: four **tool blocks** (master left/right, slave left/right). Each
block has 19 fields: tip **x,y,z** (3), rotation matrix **R** (9),
translational velocity (3), rotational velocity (3), **gripper** angle (1).

| Block (1-based column range) | Contents |
|------------------------------|----------|
| 1–19 | Master left |
| 20–38 | Master right |
| 39–57 | Slave left |
| 58–76 | Slave right |

Optional training flag **`--append_motion_features`** concatenates four
**derived** scalars per frame (velocity magnitude, smoothed velocity,
acceleration, jerk from master-left translational velocity); see
[`motion_columns.py`](src/suturing_pipeline/kinematics/motion_columns.py).

### Non-kinematic conditioning (optional)

**`--num_semantic_tokens`** adds learnable soft-prompt tokens (concatenated in
cross-attention). **`--clip_scene_prompt`** injects a single token derived
from the frozen SD **CLIP text encoder** (mean-pooled hidden states + learned
projection) so the model can pick up needle/thread/scene semantics alongside
kinematics.

Only the JIGSAWS `Suturing` task is supported. The other JIGSAWS tasks
do not ship aligned kinematics usable by this pipeline.

## What Actually Runs

The real training + evaluation loop lives under `scripts/` and
`src/suturing_pipeline/`. The original notebook-based "phase" layout
(detection, kinematics, dashboard) is still in-tree as secondary
tooling, but the main pipeline is the SD-generation stack described
below.

### Core pipeline (SD generation)

| Stage                    | Script                           | What it does |
| ------------------------ | -------------------------------- | ------------ |
| Pull JIGSAWS from Drive  | `scripts/download_jigsaws.py`    | Mirrors the JIGSAWS Drive folder (kinematics, transcriptions, meta files, `Experimental_setup/`, videos) into `data/gdrive_cache/`. |
| Train                    | `scripts/train_sd.py`            | VAE-encode frames → add noise → predict noise with LoRA-adapted U-Net conditioned on `KinematicEncoder(kin, gesture)`. Saves `step_*.pt` checkpoints with LoRA weights, encoder weights, scaler params, and `gesture_to_int` mapping. |
| Single-frame inference   | `scripts/inference_sd.py`        | Loads a checkpoint and generates one PNG for a given kinematics row + gesture id. |
| Real-vs-generated grid   | `scripts/generate_eval_grid.py`  | Balanced sample across gestures on the held-out split → `eval_grid.png` (real \| generated) + optional `gesture_sweep.png` (fixed kinematics, varied gesture). |
| Real-vs-generated video  | `scripts/generate_eval_video.py` | Walks a contiguous range of frames from one held-out trial and writes `real.mp4`, `generated.mp4`, `sidebyside.mp4`. |
| Metrics on a grid image  | `scripts/metrics_on_grid.py`     | PSNR / SSIM / histogram / edge-IoU on the exported grid, including cross-pair baselines to check the model is tracking conditioning rather than scene average. |
| Metrics on a clip pair   | `scripts/video_quality_metrics.py` | SSIM + Farneback optical-flow motion profile, flags flicker/jump windows with plain-language reasons. |
| Fréchet Inception Distance | `scripts/compute_fid.py`           | Distribution distance between two folders of PNGs (requires `pytorch-fid`). Use aligned real vs generated image sets from the same protocol. |
| Interactive comparison   | `scripts/streamlit_compare.py`   | Streamlit UI: pick a checkpoint + trial + clip length, shells out to `generate_eval_video.py`, embeds the resulting MP4s and the metrics board. |

Optional **gesture-aligned narration** (template, Ollama, or Hugging Face **text** → **Bark** TTS → ffmpeg mux) is explained for users in [docs/audio_generation.md](docs/audio_generation.md); pipeline and file layout details are in [docs/narration_audio.md](docs/narration_audio.md).

### Diagnostics

These answer *"is the pipeline broken, or is it just undertrained?"*
before burning GPU hours.

- `scripts/diagnose_vae.py` — runs real frames through the frozen SD
  VAE (encode → decode) and reports per-image PSNR. This is the
  **ceiling** of the stack; the generator cannot exceed VAE fidelity
  without also fine-tuning the VAE on surgical frames.
- `scripts/overfit_one_frame.py` — trains a fresh LoRA +
  `KinematicEncoder` on a single `(frame, kinematics, gesture)` tuple
  and snapshots generation over training. If this cannot memorise one
  frame, the wiring is wrong; if it can, the "messy output" problem is
  about data scale / steps / resolution, not the loop.

Output artefacts land under `outputs/diagnostics/...`.

### Secondary tooling (not part of the generation loop)

These scripts pre-date the SD pipeline and are retained for
exploratory / labeling work. They are **not** required for training or
evaluating the generator.

- `scripts/prepare_trials.py` — trial discovery and modality linking
  into `outputs/trial_index.csv`.
- `scripts/prepare_labeling_set.py`, `scripts/train_yolo.py`,
  `scripts/run_detection.py` — YOLO labeling bootstrap + detector
  training + detection export.
- `scripts/annotate_boxes.py` + `annotation_ui/` — instrument box
  annotation with kinematics-projected pre-labels; exports YOLO labels
  for `train_yolo.py` (see below).
- `scripts/compute_kinematics.py` — velocity / acceleration / jerk
  features.
- `scripts/run_synthesis.py`, `scripts/launch_dashboard.py` — original
  ControlNet/SVD scaffolding and comparison dashboard from the earlier
  notebook baseline.

## Repository Layout

```
src/suturing_pipeline/
  data/
    jigsaws_dataset.py        # PyTorch Dataset: lazy video frames + aligned kin + gesture
    data_utils.py             # kinematics / transcription / split parsing
    loader.py                 # Google Drive OAuth + cache materialisation
    alignment.py              # modality alignment helpers
  synthesis/
    kinematic_encoder.py      # kin + gesture -> cross-attn (optional sem + CLIP token)
    sd_sampler.py             # load ckpt once, call .sample() per frame
    controlnet_pipeline.py    # (legacy scaffold)
  detection/                  # YOLO labeling + training + export (secondary)
  annotation/                 # box annotation: kinematics pre-labels, session metrics, YOLO export
  kinematics/                 # feature engineering; motion_columns (append to kin)
  sequence/                   # temporal model components (secondary)
  dashboard/                  # original comparison dashboard (secondary)
scripts/                      # CLI entry points for every stage above
notebooks/
  colab_train_sd.ipynb        # GPU training on Colab
  01_..04_*.ipynb             # earlier phase notebooks
checkpoints/                  # LoRA checkpoints (e.g. suturing_expert_lora/step_1480.pt)
data/gdrive_cache/            # materialised JIGSAWS mirror
outputs/
  eval/                       # real-vs-generated grids
  eval_video/                 # clip comparisons + streamlit runs
  diagnostics/                # VAE ceiling + one-frame overfit
configs/base.yaml             # ingestion + legacy detection / kinematics config
```

Immutable snapshots of the original notebook are kept under
`versions/`.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1. Pull JIGSAWS into the local cache

The project reads JIGSAWS through a Google Drive OAuth client. Drop
your `google_oauth_client.json` under `./secrets/` (see
[Google Drive OAuth](#google-drive-oauth-setup) below), then:

```bash
python scripts/download_jigsaws.py
```

This mirrors the Drive folder into `data/gdrive_cache/` so every other
script sees normal file paths.

### 2. Train

Recommended (Colab / GPU). Notebook: `notebooks/colab_train_sd.ipynb`.
CLI form:

```bash
python scripts/train_sd.py \
  --data_root ./data/gdrive_cache \
  --expert_only \
  --train_mode lora \
  --image_size 256 \
  --capture 1 \
  --frame_stride 30 \
  --batch_size 4 \
  --lr 1e-4 \
  --epochs 50 \
  --save_every 500 \
  --save_dir ./checkpoints/suturing_expert_lora

# Optional (professor feedback): richer motion input + semantic conditioning
# python scripts/train_sd.py ... --append_motion_features \\
#   --num_semantic_tokens 4 \\
#   --clip_scene_prompt "robotic laparoscopic suturing needle suture thread tissue"
```

Each saved checkpoint bundles LoRA state, `KinematicEncoder` weights,
the `StandardScaler` params used at training time, and the
`gesture_to_int` vocabulary, so evaluation scripts are fully
self-contained given a single `.pt` file.

### 3. Evaluate

```bash
python scripts/generate_eval_grid.py \
  --checkpoint checkpoints/suturing_expert_lora/step_1480.pt \
  --data_root  ./data/gdrive_cache \
  --num_samples 12 --num_inference_steps 25

python scripts/generate_eval_video.py \
  --checkpoint checkpoints/suturing_expert_lora/step_1480.pt \
  --data_root  ./data/gdrive_cache \
  --num_frames 60 --frame_step 6 --num_inference_steps 20

# Optional: FID between two folders of PNGs (e.g. cropped reals vs gens)
# python scripts/compute_fid.py --path1 ./real_pngs --path2 ./gen_pngs --device cuda
```

Or launch the interactive comparison app:

```bash
streamlit run scripts/streamlit_compare.py
```

### 4. Sanity-check the stack

```bash
python scripts/diagnose_vae.py \
  --data_root ./data/gdrive_cache --num_samples 6 --image_size 256

python scripts/overfit_one_frame.py \
  --data_root ./data/gdrive_cache --steps 500 \
  --snapshot_steps 50,100,250,500
```

## Design Notes

**Why replace the text encoder.** JIGSAWS kinematics are dense,
continuous, and have a fixed 76-dim layout per timestep. Routing them
through CLIP text tokens would lose precision and waste the cross-attn
bandwidth. The `KinematicEncoder` is a 3-layer MLP that produces a
full `[77, 768]` sequence directly; the gesture label is added to
token 0 and the result is layer-normed.

**Why LoRA on cross-attn.** The base SD weights already know how to
turn latents into images; we only need to re-route *what* the U-Net
attends to. LoRA on `to_q/to_k/to_v/to_out.0` keeps ~1% of parameters
trainable and gives per-trial checkpoints small enough to ship inside
a `.pt` file alongside encoder + scaler + vocab.

**Why the VAE stays frozen.** The VAE reconstruction ceiling
(`diagnose_vae.py`) shows that the frozen SD 1.5 VAE can already round-
trip JIGSAWS suturing frames at roughly PSNR ~24–27 dB, which is above
what the current generator reaches. Fine-tuning the VAE is on the
roadmap but is not the limiting factor today.

**Temporal consistency.** Training is per-frame; there is no temporal
loss. `generate_eval_video.py` therefore uses a **fixed-seed** noise
latent across frames by default, which empirically gives the most
coherent playback. A diffusion-level temporal prior (e.g. AnimateDiff
or SVD) is the obvious next step.

## Google Drive OAuth Setup

1. Place your OAuth client secret at
   `./secrets/google_oauth_client.json`.
2. Keep `configs/base.yaml` as-is, or set:
   - `ingestion.source: gdrive`
   - `ingestion.gdrive.credentials_path: ./secrets/google_oauth_client.json`
   - `ingestion.gdrive.oauth_flow: console` (prints consent URL in
     terminal; no auto-open)
   - `ingestion.gdrive.root_folder_id: <JIGSAWS folder id>`
   - `ingestion.gdrive.materialize_local: true` to cache discovered
     files locally
3. Run `python scripts/download_jigsaws.py` (or `prepare_trials.py` if
   you want an indexed CSV of trials).

The first run prints a Google consent URL; open it manually. The
refresh token is saved to `./secrets/google_token.json` and reused on
later runs. `ingestion.source: local` is still honoured if you mirror
JIGSAWS to disk yourself.

## Data Expectations

Each Suturing trial must provide:

- `capture1.avi` and/or `capture2.avi`
- JIGSAWS kinematics file sampled at 30 Hz
- transcription / gesture timing file
- the standard `Experimental_setup/.../onetrialout/` split files
- `meta_file_Suturing.txt` — tab- or space-delimited columns (1-based):
  (1) trial / filename, (2) self-reported skill `N`/`I`/`E`, (3) GRS skill,
  (4–9) six modified GRS element scores (tissue, suture/needle, time & motion,
  flow, overall, quality of final product). The pipeline and `--expert_only` use
  **column 2** (self-reported) for expert filtering, not column 3. Full
  reference: `src/suturing_pipeline/data/jigsaws_metafile_layout.py`.

`JIGSAWSDataset` lazily opens video via OpenCV, normalises kinematics
with a `StandardScaler` fitted at init, and serialises scaler +
gesture vocab alongside every checkpoint.

## Current Status

Working:

- End-to-end kinematic-conditioned frame generation for JIGSAWS
  Suturing.
- LoRA training on Colab GPU, inference on CUDA / MPS / CPU.
- Real-vs-generated grids, gesture sweeps, clip comparisons, metrics
  board, Streamlit demo.
- VAE-ceiling and one-frame-overfit diagnostics.

Not yet / scaffolded only:

- ControlNet and SVD hookups (`scripts/run_synthesis.py`,
  `src/suturing_pipeline/synthesis/controlnet_pipeline.py`) are stubs
  from the earlier baseline and are not wired to the trained
  checkpoints.
- YOLO detector branch is independent of the generator and is kept for
  downstream analysis.
- No temporal loss during training; multi-frame coherence relies on
  fixed-seed sampling.

## Inference optimization: KinematicEncoder → ONNX → INT8 (September 26, 2026)

The encoder's last layer projects 512 features to 77 × 768 conditioning tokens, so the module carries
**30.5M parameters**. `scripts/optimize_encoder.py` exports it to ONNX (opset 17, dynamic batch),
checks the graph, applies dynamic INT8 weight quantization with ONNX Runtime, and benchmarks on CPU (arm64 CPU, 4 threads):

| | Size | Batch 1 p50 | Batch 16 p50 | Output vs PyTorch |
|---|---|---|---|---|
| PyTorch FP32 | – | 2.007 ms | 4.649 ms | reference |
| ONNX Runtime FP32 | 116.4 MB | 1.912 ms | 4.638 ms | max abs error 5e-06 |
| ONNX Runtime INT8 | 29.3 MB | 0.429 ms | 1.996 ms | cosine ≥ 0.999858 |

INT8 makes the encoder **4.0x smaller** and **4.7x faster at batch 1**
(2.3x at batch 16). The trained checkpoint lives on Colab, so these runs use
seeded weights: size and latency are properties of the architecture, while output fidelity must be re-checked with
`--checkpoint` on the trained encoder before relying on INT8 in generation. The U-Net (the dominant cost) and TensorRT
on an NVIDIA GPU are the next step and were not run here. `tests/test_encoder_export.py` checks export parity and INT8
fidelity on a small encoder.

## Instrument box annotation with kinematics pre-labels (September 28, 2026)

The YOLO branch above needs instrument boxes, and drawing them by hand is slow. JIGSAWS records the robot's
kinematics for every video frame but ships no camera calibration. `scripts/annotate_boxes.py` uses the
kinematics anyway: after an annotator has boxed a few frames of a trial, it learns where each instrument's
jaw appears in the image from the tool-tip position and shaft direction, then pre-labels the remaining
frames. This is the same idea driving datasets use when they project LiDAR points into camera images.

- **Model** (`src/suturing_pipeline/annotation/projection.py`). For each instrument, the box centre is a linear
  function of the tool-tip position (the shaft axis becomes a candidate once 7 frames are labeled), with a per-instrument
  offset. The box size is the median of the annotator's boxes. Models are compared by leave-one-frame-out
  error, and the UI shows that error ("about N px off on held-out frames"). The fit is per trial and camera,
  because the camera can move between sessions, and it reruns after every saved frame. In these videos the
  kinematics' "slave left" arm appears on the right of the image, and the mapping accounts for that.
- **Measurement** (`session.py`). Every pre-label shown is stored next to the box the annotator saved. The
  quality panel reports the share of pre-labels kept (IoU ≥ 0.5), the median IoU and centre error, the
  boxes deleted, and the median seconds per frame with and without pre-labels.
- **UI** (`annotation_ui/`, React + TypeScript on a canvas). Drag to draw; drag inside a box to move it and
  a corner to resize; `1`/`2` choose the instrument; `Delete` removes a box; `Enter` saves and loads the
  next frame. Pre-labels are dashed until edited.
- **Export.** Saved boxes are written as `images/<split>/` and `labels/<split>/` in YOLO format (classes
  `left_tool`, `right_tool`), the layout `scripts/train_yolo.py` expects. Splits are assigned per trial
  with the same hash as `prepare_labeling_set.py`.

```bash
python scripts/annotate_boxes.py --jigsaws /path/to/Suturing --trials B001,C001 --db outputs/annotation/boxes.db
python scripts/annotate_boxes.py --demo          # synthetic frames with a known camera, no dataset needed
cd annotation_ui && npm install && npm run build # UI at http://127.0.0.1:8780
```

Tests: `tests/test_box_annotation.py` covers the projection recovering a known camera, model selection,
off-screen handling, acceptance and timing metrics, validation, YOLO export, and frame/kinematics pairing
on a generated video. `annotation_ui/src/geometry.test.ts` covers box geometry. `annotation_ui/e2e/`
has a Playwright flow on Chromium, Firefox and WebKit: it draws boxes with the mouse, accepts pre-labels
with the keyboard, moves, resizes and deletes a box, and exports. The `Box annotation` workflow runs
all of it in CI.
