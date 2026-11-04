# Inference

Run commands from the repository root, in the corresponding
[model environment](../docs/SETUP.md). Prepare the original backbone and a
packed NanoKV codec. Fast additionally requires reader weights and a manifest.

The commands below use the published-weight profiles. For **your own trained
weights**, first create a local profile following the
[artifact guide](../configs/README.md#locally-trained-artifacts), then replace
the `--profile` path below with that absolute JSON path. Do not put newly fitted
weights behind a published-weight checksum.

Use absolute filesystem paths for model weights, inputs, and outputs. The MG
`--config configs/...yaml` path is an intentional exception: it identifies a
configuration shipped inside the native MG integration. Choose a fresh output
path for each run. Optional `CUDA_VISIBLE_DEVICES=1` before `python` selects GPU 1.

## HY-WorldPlay

```bash
export ARTIFACTS=/absolute/path/to/hy/artifacts
python infer/run.py hy generate --profile configs/hy/nanokv.json -- \
  --model_path /absolute/path/to/HunyuanVideo-1.5 \
  --action_ckpt /absolute/path/to/action_model/model.safetensors \
  --image_path /absolute/path/to/input.png --prompt "Your prompt" \
  --pose left-31,right-32 --output_path /absolute/path/to/output
```

For Fast, replace `generate` with `generate_fast` and use `configs/hy/fast.json`.
The profile uses the paper's 253-frame, four-step, rank-256 configuration.
The generated video is written to `output_path/gen.mp4`.
For a locally fitted Fast reader, use the `fast-local.json` profile produced by
the training guide with `generate_fast`.

## LingBot-World 2.0

```bash
export ARTIFACTS=/absolute/path/to/lingbot/artifacts
python infer/run.py lingbot generate --profile configs/lingbot/nanokv.json -- \
  --ckpt_dir /absolute/path/to/lingbot-backbone \
  --image /absolute/path/to/input.png --prompt "Your prompt" \
  --action_path /absolute/path/to/actions \
  --save_file /absolute/path/to/output.mp4
```

For Fast, use `generate_fast` with `configs/lingbot/fast.json`.
The profile uses 125 frames and the causal-fast backbone.

LingBot profiles fix `OMP_NUM_THREADS=56` and `MKL_NUM_THREADS=56`, matching
the reference runs' CPU text-encoder setting. Use the same setting across
Original, NanoKV and Fast when comparing their generated videos.

The action directory contains `poses.npy` (camera-to-world matrices, OpenCV
coordinates, shape `[T,4,4]`) and `intrinsics.npy` (shape `[T,4]`,
`fx,fy,cx,cy` for an 832×480 image). Supply at least 125 poses for this profile;
the native model shortens generation when fewer poses are provided.

## Matrix-Game 3.5

```bash
export ARTIFACTS=/absolute/path/to/mg/artifacts
export NANOKV_RUN_CACHE=/absolute/path/to/mg_runtime_cache
python infer/run.py mg generate --profile configs/mg/nanokv.json -- \
  --config configs/infer_distilled_2blocks_retrace.yaml \
  --checkpoint /absolute/path/to/first-person.safetensors \
  --wan-dir /absolute/path/to/Wan2.2-TI2V-5B \
  --tokenizer-dir /absolute/path/to/Wan2.2-TI2V-5B/google/umt5-xxl \
  --da3-dir /absolute/path/to/DA3NESTED-GIANT-LARGE-1.1 \
  --image /absolute/path/to/input.png --camera /absolute/path/to/camera.npz \
  --prompt "Your prompt" --output /absolute/path/to/output.mp4
```

For Fast, use `generate_fast` with `configs/mg/fast.json`. Run the native help to
set backbone component locations (`--wan-dir`, `--tokenizer-dir`, `--da3-dir`)
for your installation:

```bash
python infer/run.py mg generate -- --help
```

Set `NANOKV_RUN_CACHE` to an absolute scratch directory to store MG's temporary
depth, camera and memory-latent files outside the checkout.

The camera archive contains `extrinsics_c2w` with shape `[T,4,4]` and
`intrinsics` with shape `[T,4]`. The two-block configuration uses 169 frames.

## Baselines

Each model has `original.json`, `qvg.json`, `turboquant.json`, and `kvtc.json`
profiles. Use these with `generate` and the same model/input arguments as above.
Profiles reset model experiment switches before applying their fixed settings.
For KVTC, set `KVTC_ARTIFACTS` to its fitted artifacts. For QVG, install the
official implementation and set its absolute checkout location:

```bash
git clone https://github.com/svg-project/Quant-VideoGen.git external/Quant-VideoGen
export QVG_ROOT="$(pwd)/external/Quant-VideoGen"
```

The numerical budgets are recorded in `configs/paper_settings.json`.
`--dry-run` before `--` displays the command without loading a model.

## Video metrics

Use identical prompts, input images, actions, seeds, frame counts and resolution
for Original and compressed runs. Install `lpips` and `scikit-image` in the
evaluation environment, then run:

```bash
python eval/video_pair.py --reference /absolute/path/to/original.mp4 \
  --candidate /absolute/path/to/nanokv.mp4 --output /absolute/path/to/metrics.json
```

The output includes per-frame PSNR/SSIM/LPIPS and both all-frame and
generated-frame means (excluding the shared input frame).
