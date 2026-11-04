# Train Matrix-Game 3.5

Activate the MG environment and run from the repository root. This guide fits
rank-384 K/V representations, 64 shared centers, and INT4 coefficients.

## 1. Prepare paths and fitting inputs

```bash
export WORK=/absolute/path/to/mg_training
export WAN=/absolute/path/to/Wan2.2-TI2V-5B
export CHECKPOINT=/absolute/path/to/first-person.safetensors
export DA3=/absolute/path/to/DA3NESTED-GIANT-LARGE-1.1
export NANOKV_RUN_CACHE="$WORK/runtime_cache"
mkdir -p "$WORK"
```

Prepare 48 training and four validation trajectories with image-disjoint
splits. Each trajectory needs an input image, prompt, and a camera archive
in the [inference format](../infer/README.md#matrix-game-35).

## 2. Collect paired raw/residual samples

For each training trajectory, run the following command with its own case ID,
image, prompt, camera, and output video:

```bash
python train/run.py mg collect --profile configs/mg/original.json \
  --env MG35_KVCODEC_MODE=collect \
  --env MG35_KVCODEC_OUTPUT="$WORK/samples/train/case_000" \
  --env MG35_KVCODEC_MAX_ROWS=4096 \
  --env MG35_KVCODEC_ROWS_PER_WRITE=1024 \
  --env MG35_KVCODEC_COLLECT_WRITE_IDS=0,2,4,6 \
  --env MG35_KVCODEC_COLLECT_C64=1 \
  --env MG35_KVCODEC_CENTER_STORAGE=bf16 \
  --env MG35_KVCODEC_CENTER_BLOCK_SIZE=64 -- \
  --config configs/infer_nanokv.yaml \
  --checkpoint "$CHECKPOINT" --wan-dir "$WAN" \
  --tokenizer-dir "$WAN/google/umt5-xxl" --da3-dir "$DA3" \
  --image /absolute/path/to/train.png --camera /absolute/path/to/train_camera.npz \
  --prompt "A mountain landscape" --output "$WORK/train_000.mp4"
```

Repeat for validation under `$WORK/samples/val/case_XXX`. Each case directory
contains `k/`, `v/`, `k_c64/`, and `v_c64/`, each with 30 layer files.
Collection stores token-matched raw/residual rows; the sample identifiers must
match across all four targets.

Validate the actual files and create fitting manifests:

```bash
python train/mg_manifest.py --split train --roots "$WORK"/samples/train/case_* \
  --output "$WORK/train_manifest.json"
python train/mg_manifest.py --split val --roots "$WORK"/samples/val/case_* \
  --output "$WORK/val_manifest.json"
```

The validator checks all layer files, their shapes, paired identifiers, and
center metadata. It refuses incomplete collection rather than marking it done.

## 3. Fit the representation and Value codec

Build repeated command arguments from the collected case directories:

```bash
train_roots=("$WORK"/samples/train/case_*)
val_roots=("$WORK"/samples/val/case_*)
pca_args=()
train_args=()
val_args=()
for root in "${train_roots[@]}"; do
  pca_args+=(--samples "$root")
  train_args+=(--train "$root")
done
for root in "${val_roots[@]}"; do val_args+=(--val "$root"); done

python train/run.py mg fit_basis -- \
  "${pca_args[@]}" --output "$WORK/basis" --rank 384 \
  --targets k_c64 v_c64 --center-storage bf16 --center-block-size 64 \
  --expected-roots "${#train_roots[@]}" --expected-rows-per-root 4096 \
  --expected-total-rows "$((4096 * ${#train_roots[@]}))" \
  --collection-manifest "$WORK/train_manifest.json"

for layer in $(seq 0 29); do
  python train/run.py mg fit_value -- \
    "${train_args[@]}" "${val_args[@]}" \
    --basis-dir "$WORK/basis" --output-dir "$WORK/v_qat" --layer "$layer" \
    --rank 384 --steps 12000 --batch-size 512 --lr 1e-4 \
    --center-storage bf16 --center-block-size 64 \
    --train-manifest "$WORK/train_manifest.json" --val-manifest "$WORK/val_manifest.json" \
    --expected-train-roots "${#train_roots[@]}" --expected-val-roots "${#val_roots[@]}" \
    --expected-rows-per-root 4096 \
    --expected-train-rows "$((4096 * ${#train_roots[@]}))" \
    --expected-val-rows "$((4096 * ${#val_roots[@]}))"
done
```

The MG deployment uses the residual PCA Key decoder and the fitted Value
codec. Export this combination with the `c64_vqat` rung.

## 4. Export and run

```bash
export ARTIFACTS="$WORK/artifacts"
python train/run.py mg pack_codec -- \
  --basis-dir "$WORK/basis" --v-qat-dir "$WORK/v_qat" --rung c64_vqat \
  --rank 384 --num-layers 30 --center-storage bf16 --center-block-size 64 \
  --output "$ARTIFACTS/codec.pt"
python -m nanokv.prepare_profile --model mg --artifacts "$ARTIFACTS" \
  --output "$WORK/nanokv-local.json"
```

Use the local profile in the [inference command](../infer/README.md#matrix-game-35).
Continue with [Fast reader fitting](fast.md) for the optional attention path.
