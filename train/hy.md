# Train HY-WorldPlay

Activate the HY environment and run from the repository root. This guide uses
rank 256 (the 32× dimensional/bitwidth setting); the intermediate residual PCA
bases retain 512 components. Use absolute paths throughout.

## 1. Prepare fitting cases and paths

```bash
export WORK=/absolute/path/to/hy_training
export BASE=/absolute/path/to/HunyuanVideo-1.5
export ACTION=/absolute/path/to/action_model/model.safetensors
mkdir -p "$WORK"
```

Create `$WORK/cases.json` with image-disjoint `train` and `val` lists.
Each list contains objects of this form:

```json
{
  "train": [
    {"id": "train_000", "image": "/absolute/path/to/train.png",
     "prompt": "A mountain landscape", "pose": "left-31,right-32"}
  ],
  "val": [
    {"id": "val_000", "image": "/absolute/path/to/val.png",
     "prompt": "A coastal landscape", "pose": "left-31,right-32"}
  ],
  "test": []
}
```

Use your complete fitting split, not just the two illustrative entries. Keep
evaluation images out of these lists. The collector's case indices refer to
the concatenated train, val, test lists, in that order.

## 2. Collect paired KV samples

For example, with 48 training and four validation trajectories:

```bash
python train/run.py hy collect -- \
  --split_json "$WORK/cases.json" --cases 0-51 \
  --positions 1 3 5 7 --layers 0-53 --target kv --storage_format int8 \
  --disable_model_offload --optimized_cache_lifecycle \
  --model_path "$BASE" --action_ckpt "$ACTION" \
  --output_dir "$WORK/samples"
```

Adjust `--cases` to your actual manifest length. Outputs are organized as
`samples/{train,val}/CASE/position_XX/kv/*.pt`. Each trajectory supplies four
sample positions. Both fitting stages below read this same split-aware root.

## 3. Fit residual bases and the Value codec

```bash
for layer in $(seq 0 53); do
  tag=$(printf '%02d' "$layer")
  for target in k v; do
    python train/run.py hy fit_basis -- \
      --data-root "$WORK/samples" --target "$target" --layer "$layer" --rank 512 \
      --output "$WORK/${target}_basis/layer_$tag/residual_pca512.pt"
  done
  python train/run.py hy fit_value -- \
    --data-root "$WORK/samples" --target v --layer "$layer" --rank 256 \
    --clusters 64 --kmeans-iters 2 --rows-per-atom 1024 \
    --steps 12000 --batch-size 512 --eval-every 200 --lr 1e-4 \
    --initialization random_orthogonal \
    --residual-basis "$WORK/v_basis/layer_$tag/residual_pca512.pt" \
    --output-dir "$WORK/v_qat/layer_$tag"
done
python train/run.py hy export_key -- \
  --source "$WORK/k_basis" --output-root "$WORK/k_export"
python train/run.py hy export_value -- \
  --source "$WORK/v_qat" --output-root "$WORK/v_export"
```

The Value trainer selects `checkpoint_best.pt` using validation reconstruction
error. Export creates `k_export/basis`, `v_export/basis`, and
`v_export/linear`. K and V bases must remain in separate directories.

## 4. Fit the Key decoder

Write the training list, without the outer split object, for online alignment:

```bash
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ["WORK"])
cases = json.loads((root / "cases.json").read_text())
(root / "key_cases.json").write_text(json.dumps(cases["train"], indent=2))
PY
python train/run.py hy fit_key -- \
  --manifest "$WORK/key_cases.json" --basis-root "$WORK/k_export/basis" \
  --model-path "$BASE" --action-ckpt "$ACTION" \
  --rank 256 --output-root "$WORK/k_alignment"
```

This stage initializes decoder deltas and aligns groups of nine layers in
sequence, using five updates at learning rate `5e-6` on chunks 1–4.
Use native `--resume` to resume the recorded fitting state. The output contains
`layer_XX/checkpoints/checkpoint_functional_ft.pt`.

## 5. Export and run

```bash
export ARTIFACTS="$WORK/artifacts"
python train/run.py hy pack_codec -- \
  --k-basis "$WORK/k_export/basis" --k-linear "$WORK/k_alignment" \
  --v-basis "$WORK/v_export/basis" --v-linear "$WORK/v_export/linear" \
  --rank 256 --layers 54 --output "$ARTIFACTS/codec.pt"
python -m nanokv.prepare_profile --model hy --artifacts "$ARTIFACTS" \
  --output "$WORK/nanokv-local.json"
```

Use `$WORK/nanokv-local.json` instead of `configs/hy/nanokv.json` in the
[inference command](../infer/README.md#hy-worldplay). The local profile binds
the checksum of your newly trained operator. Continue with
[Fast reader fitting](fast.md) if needed.
