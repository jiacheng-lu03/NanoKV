# Train LingBot-World 2.0

Activate the LingBot environment and run from the repository root. The deployed
codec rank is 640, with 64 shared centers and two clustering iterations.

## 1. Prepare paths and splits

```bash
export WORK=/absolute/path/to/lingbot_training
export BASE=/absolute/path/to/lingbot-world-v2-14b-causal-fast
mkdir -p "$WORK"
```

Use 48 training trajectories and four development trajectories for the supplied
sample-count gates (four collected chunks each: 192/16 samples per layer).
Split by input image, keeping every action trajectory from an image in one
split. Action files follow the [inference format](../infer/README.md#lingbot-world-20).

## 2. Collect both splits

Run this command for each training trajectory, changing the image, prompt,
action directory, case ID, and output video:

```bash
python train/run.py lingbot collect --profile configs/lingbot/original.json -- \
  --ckpt_dir "$BASE" --image /absolute/path/to/train.png \
  --prompt "A mountain landscape" --action_path /absolute/path/to/actions \
  --save_file "$WORK/train_000.mp4" \
  --kv_dump_root "$WORK/samples" --kv_dump_split train \
  --kv_dump_case_id train_000 --kv_dump_chunks 1,3,5,7 --kv_dump_tokens 1024
```

For each development trajectory, use `--kv_dump_split dev`, a unique
`dev_XXX` case ID, and its own output video. The result is
`samples/{train,dev}/CASE/chunk_XX/rank_00/layer_XX_{k,v}.pt`.
Keep all requested chunks and layers; fitting validates sample counts.

## 3. Fit residual bases and Values

```bash
for layer in $(seq 0 39); do
  for target in k v; do
    python train/run.py lingbot fit_basis -- \
      --data-root "$WORK/samples" --output-root "$WORK/${target}_pca" \
      --target "$target" --layer "$layer" --rank 640 \
      --expected-train-atoms 192 --expected-dev-atoms 16
  done
  python train/run.py lingbot fit_value -- \
    --data-root "$WORK/samples" --pca-root "$WORK/v_pca" \
    --output-root "$WORK/v_qat" --layer "$layer" --rank 640 \
    --steps 12000 --batch-size 512 --lr 1e-4 \
    --expected-train-atoms 192 --expected-dev-atoms 16
done
```

Each output root contains `rank_00/layer_XX/`. If you intentionally use a
different fitting split, change both sample-count gates to its actual size.

## 4. Fit Keys

Create `$WORK/key_cases.json` with all 48 training trajectories:

```json
{
  "total_chunks": 8,
  "sampled_commits": [2, 4, 6, 8],
  "train": [
    {"id": "train_000", "image": "/absolute/path/to/train.png",
     "prompt": "A mountain landscape",
     "action_path": "/absolute/path/to/actions", "frame_num": 125}
  ]
}
```

The example shows one entry; populate the complete list. Alignment uses the
same training images as collection, but updates on chunks 2,4,6,8.

```bash
python train/run.py lingbot fit_key -- \
  --profile full48_normalized --gpu 1 \
  --manifest "$WORK/key_cases.json" --artifact-root "$WORK" \
  --weights "$BASE" --output-root "$WORK/k_alignment"
```

Here `--gpu 1` selects physical GPU 1. The trainer reads
`$WORK/k_pca`, `$WORK/v_pca`, and `$WORK/v_qat`, fits five groups of eight
layers, and merges the result into `k_alignment/k_a1_final.pt`.
Its learning rate is `2.5e-6`, with one update at the selected denoising step.
Use `--audit-only` for an input/protocol check before loading the model.

## 5. Export and run

```bash
export ARTIFACTS="$WORK/artifacts"
python train/run.py lingbot pack_codec -- \
  --k-basis "$WORK/k_pca" --v-basis "$WORK/v_pca" --v-qat "$WORK/v_qat" \
  --k-a1 "$WORK/k_alignment/k_a1_final.pt" \
  --rank 640 --layers 40 --sp-ranks 1 --output "$ARTIFACTS/codec.pt"
python -m nanokv.prepare_profile --model lingbot --artifacts "$ARTIFACTS" \
  --output "$WORK/nanokv-local.json"
```

Use the local profile in the [inference command](../infer/README.md#lingbot-world-20).
Then follow [Fast reader fitting](fast.md) to construct mixed-dimensional readers.
