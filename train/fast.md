# Fit NanoKV-Fast readers

Start with a fitted `codec.pt` from the model's
[codec guide](README.md). Keep the codec frozen while fitting readers.
Use the corresponding model environment and run from the repository root.

Set `WORK` to the training directory and `ARTIFACTS=$WORK/artifacts`.
The commands below use the `$WORK/nanokv-local.json` profile created at the end
of codec training. If starting from released weights, first create that profile
with `python -m nanokv.prepare_profile`.

Use development images for fitting and selecting readers; reserve evaluation
images for final comparison. Capture creates attention samples and can consume
substantial disk space. Keep each trajectory in its own directory where noted.

## HY-WorldPlay

### Capture attention samples

With `BASE` and `ACTION` defined as in [HY training](hy.md), run once per
training trajectory:

```bash
python train/run.py hy collect_fast --profile "$WORK/nanokv-local.json" \
  --env HY_CODEC_NATIVE_MIXED_DUMP_ENABLE=1 \
  --env HY_CODEC_NATIVE_MIXED_DUMP_DIR="$WORK/reader_samples" \
  --env HY_CODEC_NATIVE_MIXED_DUMP_LAYERS=all \
  --env HY_CODEC_NATIVE_MIXED_DUMP_AR_CHUNKS=2,4,6 \
  --env HY_CODEC_NATIVE_MIXED_DUMP_FULL_C64=1 \
  --env HY_KV_AE3D_CASE_ID=train_000 --env HY_KV_AE3D_SPLIT=train -- \
  --model_path "$BASE" --action_ckpt "$ACTION" \
  --image_path /absolute/path/to/train.png --prompt "A mountain landscape" \
  --pose left-31,right-32 --output_path "$WORK/reader_video_train_000"
```

Use unique `train_XXX` IDs. For held-out development images, use `dev_XXX`
IDs and `HY_KV_AE3D_SPLIT=dev`, with the same sample directory. The files are
`CASE_chunkXX_layerYY_mixed.pt`.

### Fit the required layer/width combinations

The allocation determines which experts are needed. Generate its job list:

```bash
python - <<'PY' > "$WORK/reader_jobs.txt"
import json
allocation = json.load(open("configs/hy/head_dimensions.json"))
for layer, widths in sorted(allocation.items(), key=lambda item: int(item[0])):
    if set(widths) != {128}:
        for width in sorted(set(widths)):
            print(int(layer), width)
PY

while read -r layer width; do
  tag=$(printf '%02d' "$layer")
  python train/run.py hy fit_fast -- \
    --data-root "$WORK/reader_samples" --operator "$ARTIFACTS/codec.pt" \
    --world-model "$ACTION" --output-dir "$WORK/fit_d$width" \
    --layer "$layer" --key-dim "$width" --value-dim "$width" \
    --main-k-mode learned_rope --k-anchor-mode low --v-anchor-mode low \
    --split-mode chunk_holdout --heldout-prefix dev \
    --expected-train 36 --expected-heldout 12
  mkdir -p "$WORK/experts/layer_$tag"
  cp "$WORK/fit_d$width/layer_${tag}_k${width}_v${width}.pt" \
    "$WORK/experts/layer_$tag/d$width.pt"
done < "$WORK/reader_jobs.txt"
```

The example count gates correspond to 12 training trajectories and four
held-out development trajectories, each captured at three chunks. Change the
gates to your actual counts. Chunks 2/4 provide fitting samples; chunk 6 provides
within-trajectory validation. The separate `dev_*` images are also evaluated.
Full-dimensional fallback layers do not require reader files.

## LingBot-World 2.0

### Initialize readers and capture with the full-width reader

```bash
for width in 32 64 96 128; do
  python train/run.py lingbot init_fast -- \
    --operator "$ARTIFACTS/codec.pt" --width "$width" \
    --output "$WORK/initial/analytic_d$width"
done

python train/run.py lingbot collect_fast --profile "$WORK/nanokv-local.json" \
  --env LINGBOT_CODEC_NATIVE_ENABLE=1 \
  --env LINGBOT_CODEC_NATIVE_MANIFEST="$WORK/initial/analytic_d128/manifest.json" \
  --env LINGBOT_CODEC_NATIVE_D128_JOINT=1 \
  --env LINGBOT_CODEC_NATIVE_ALLOW_UNCALIBRATED=1 \
  --env LINGBOT_CODEC_NATIVE_DUMP_ROOT="$WORK/reader_samples/train_000_forward" \
  --env LINGBOT_CODEC_NATIVE_DUMP_CHUNKS=4,6,8 -- \
  --ckpt_dir "$BASE" --image /absolute/path/to/train.png \
  --prompt "A mountain landscape" --action_path /absolute/path/to/actions \
  --save_file "$WORK/reader_video_train_000.mp4"
```

Repeat with unique `train_NNN_ACTION` directories. This naming convention is
checked by the fitter's `--require-train-split` guard. Capture requires the
initialized **width-128** reader; a mixed-width reader is not a valid teacher.
The uncalibrated flag here permits the analytic initialization for collection.

### Fit and gather experts

```bash
for width in 32 64 96 128; do
  python train/run.py lingbot fit_fast -- \
    --initial-root "$WORK/initial" --operator "$ARTIFACTS/codec.pt" \
    --samples "$WORK/reader_samples" --output "$WORK/fit_d$width" \
    --width "$width" --center-output --analytic-only --require-train-split \
    --gain-cap 2
  for layer in $(seq 0 39); do
    tag=$(printf '%02d' "$layer")
    mkdir -p "$WORK/experts/layer_$tag"
    cp "$WORK/fit_d$width/readers/rank_00/layer_$tag.pt" \
      "$WORK/experts/layer_$tag/d$width.pt"
  done
done
```

The fitter uses chunks before 8 for fitting and chunk 8 for validation.
The resulting Value maps account for the attention output; the codec remains
frozen. The final exporter selects only the widths used by each head.

## Matrix-Game 3.5

### Capture native/history attention samples

With `CHECKPOINT`, `WAN`, `DA3`, and `NANOKV_RUN_CACHE` set as in
[MG training](mg.md), run each fitting trajectory:

```bash
python train/run.py mg collect_fast --profile "$WORK/nanokv-local.json" \
  --env MG35_READER_CAPTURE_DIR="$WORK/reader_samples/train_000" \
  --env MG35_READER_CAPTURE_SPLIT=train --env MG35_READER_CAPTURE_FULL=1 -- \
  --config configs/infer_nanokv.yaml \
  --checkpoint "$CHECKPOINT" --wan-dir "$WAN" \
  --tokenizer-dir "$WAN/google/umt5-xxl" --da3-dir "$DA3" \
  --image /absolute/path/to/train.png --camera /absolute/path/to/train_camera.npz \
  --prompt "A mountain landscape" --output "$WORK/reader_video_train_000.mp4"
```

Use a separate directory for each trajectory. The collector writes
`full_layerXX_callYYY.pt` at selected calls. The fitter requires at least three
captured files per layer and holds out the final captured call of each case.

### Fit width-64 readers and retain full-width readers

```bash
python train/run.py mg init_fast -- \
  --operator "$ARTIFACTS/codec.pt" --output "$WORK/experts"
NANOKV_CODEC="$ARTIFACTS/codec.pt" python train/run.py mg fit_fast -- \
  --samples "$WORK/reader_samples" --out "$WORK/fit_d64" --dim 64 --history-only \
  --layers "$(seq -s, 0 29)"
for layer in $(seq 0 29); do
  tag=$(printf '%02d' "$layer")
  cp "$WORK/fit_d64/layer$tag.pt" "$WORK/experts/layer_$tag/d64.pt"
done
```

Initialization creates `d128.pt` in each expert directory. The fixed MG
allocation uses only widths 64 and 128.

## Export a runnable Fast profile

After completing the appropriate model section, set `MODEL` to `hy`,
`lingbot`, or `mg`:

```bash
MODEL=hy
python train/assemble_readers.py --model "$MODEL" \
  --experts "$WORK/experts" --operator "$ARTIFACTS/codec.pt" \
  --output "$WORK/assembled_readers"
python -m nanokv.prepare_profile --model "$MODEL" --variant fast \
  --artifacts "$ARTIFACTS" --manifest "$WORK/assembled_readers/manifest.json" \
  --output "$WORK/fast-local.json"
```

The assembler writes readers and a manifest with the fixed allocation, routing
policy, and checksums. It refuses missing experts or an existing output directory.
The profile binder checks that the manifest belongs to the specified codec.

Run `infer/run.py MODEL generate_fast --profile "$WORK/fast-local.json" -- ...`
using the full model/input arguments in the [inference guide](../infer/README.md).
Compare against the same codec's NanoKV output with identical generation inputs.
