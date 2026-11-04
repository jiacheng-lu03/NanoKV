# Reproduction configurations

`paper_settings.json` records the codec, routing and baseline settings in the
paper. It is a protocol reference, not a model-native inference configuration.
Each model retains its own argument names and positional/cache implementation.

`MODEL/fast_manifest.template.json` records the final reader selection and
routing policy. After obtaining the corresponding weights, place the packed
codec at `codec.pt` and readers under `readers/` using the names in the template.
Bind the paths before inference:

```bash
python -m nanokv.prepare_artifacts --model lingbot \
  --artifacts /absolute/path/to/lingbot --output /absolute/path/to/lingbot/manifest.json
```

The binder verifies available codec/reader SHA256 checksums and refuses to
overwrite an existing manifest. A newly trained reader must use its own
exported manifest and checksums, rather than the published-weight manifest.

The published MG reader retains its original `calibrated:false` metadata;
the MG Fast profile includes its matching compatibility flag. Codec and reader
SHA256 checks remain enabled. Newly fitted readers use the manifest produced
by the reader exporter.

Hugging Face repository identifiers will be added when the weights are uploaded.
The codecs and Fast readers are separate from the original backbone checkpoints.

## Locally trained artifacts

After codec fitting, place the packed operator at `ARTIFACTS/codec.pt` and bind
the local file to the model's inference settings:

```bash
python -m nanokv.prepare_profile --model hy \
  --artifacts /absolute/path/to/artifacts \
  --output /absolute/path/to/nanokv-local.json
```

For Fast, first [fit and assemble the readers](../train/fast.md), then bind the
assembled manifest to the same codec:

```bash
python -m nanokv.prepare_profile --model hy --variant fast \
  --artifacts /absolute/path/to/artifacts \
  --manifest /absolute/path/to/assembled_readers/manifest.json \
  --output /absolute/path/to/fast-local.json
```

Use `lingbot` or `mg` for the other models. Pass the resulting absolute profile
path to `infer/run.py --profile`. The generated profiles contain absolute paths
and the actual codec SHA256; they do not require an `ARTIFACTS` environment
variable. Regenerate the profile if you move the artifacts or replace the codec.
The helper refuses to overwrite an existing profile.
