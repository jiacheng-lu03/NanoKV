# Model integrations

Each backbone has two implementations:

- `compression/`: NanoKV codec integration and standard-dimensional attention.
- `fast/`: the same compressed representation with optional mixed-dimensional attention.

Use the shared `train/run.py` and `infer/run.py` commands from the repository
root. They select the appropriate implementation and its native working directory.

## NanoKV code organization

Model-specific fitting entrypoints live in `compression/scripts/` and
`fast/scripts/`. Runtime codec and attention modules stay alongside the native
model modules so their imports and execution paths remain explicit.

| Prefix | Responsibility | Example |
|---|---|---|
| `collect_` | Collect fitting inputs | `collect_kv.py` |
| `fit_` | Fit statistical representations or readers | `fit_residual_basis.py` |
| `train_` | Optimize codec or reader parameters | `train_value_codec.py` |
| `initialize_` | Construct initial decoder/reader weights | `initialize_attention_readers.py` |
| `export_` | Convert fitted components into deployment form | `export_value_codec.py` |
| `pack_` | Assemble packed inference operators | `pack_codec.py` |
| `merge_` | Combine fitted layer groups | `merge_key_decoders.py` |

Use descriptive `snake_case` names for NanoKV files. Experiment dates, run IDs,
and temporary labels belong in local output directories, not source filenames.
Upstream model files keep their established names. Serialized weight format
identifiers and runtime environment keys retain their compatibility contracts.

See the [training overview](../train/README.md) and
[inference guide](../infer/README.md) for the supported commands.
