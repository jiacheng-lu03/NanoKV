# License scope

This repository contains NanoKV's independent original code and model
integrations distributed under different licenses. The root [MIT license](LICENSE)
applies only to the original components listed below; it does not relicense
upstream code or the repository as a whole.

## NanoKV original components — MIT

- `nanokv/`: shared launch, workflow, and artifact-binding utilities.
- `train/*.py`: independent training launch, manifest, and reader-assembly utilities.
- `infer/run.py` and `eval/video_pair.py`: independent inference and evaluation launchers.
- `configs/`: NanoKV-authored configuration files and environment specifications.
- NanoKV-authored documentation in `README.md`, `train/*.md`, `infer/README.md`,
  `configs/README.md`, and `docs/*.md`, and the project page `docs/index.html`.

A launcher's MIT license does not change the license of the model implementation
or dependency that it invokes. The figure assets in `docs/assets/`, model weights,
and datasets are not covered by this MIT grant.

## Model integrations — retained upstream terms

The model source trees under `models/`, including their adapted integration code,
are outside the root MIT grant. Their existing licenses and notices remain in
force. Separately licensed vendored components retain their own terms.

| Component | License | Included license files |
|---|---|---|
| HY-WorldPlay | Tencent HY-WorldPlay Community License Agreement | [Compression](models/HY-WorldPlay/compression/License.txt), [Fast](models/HY-WorldPlay/fast/License.txt) |
| LingBot-World 2.0 | CC BY-NC-SA 4.0 | [Compression](models/LingBot-World-2.0/compression/LICENSE.txt), [Fast](models/LingBot-World-2.0/fast/LICENSE.txt) |
| Matrix-Game 3.5 | Apache-2.0 | [Compression](models/Matrix-Game-3.5/compression/LICENSE), [Fast](models/Matrix-Game-3.5/fast/LICENSE) |

See [third-party notices](THIRD_PARTY_NOTICES.md) for included dependencies.
In particular, the MIT grant for independent NanoKV utilities does not remove
LingBot's noncommercial/share-alike terms or HY-WorldPlay's community-license
conditions. Backbone weights and separately installed dependencies remain
subject to their respective licenses.
