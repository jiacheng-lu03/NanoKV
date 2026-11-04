# Project-page video previews

These six scenes come from the NanoKV paper's teaser and appendix. Each scene
contains matched Original, NanoKV, and NanoKV-Fast outputs at the nominal 32×
configuration, selected from the existing 400-video evaluations.

- HY-WorldPlay: `test_067_left_right`, `test_021_left_right`.
- LingBot-World 2.0: `test_068_forward_backward`, `test_024_right_left`.
- Matrix-Game 3.5: `test_072_right_left`, `test_040_left_right`.

The web previews use H.264, retain all frames at the original frame rate, and
are resized to a maximum width of 960 pixels without cropping. They are silent,
pre-rendered evaluation outputs; playback speed is not generation throughput.
The images are poster frames from the corresponding NanoKV previews.

The [scene manifest](../scenes.json) records case IDs, source hashes, preview
hashes, frame counts, and frame rates. Reported paper metrics are computed from
the original evaluation videos, not these web transcodes. These media assets
are outside the MIT grant for independent NanoKV software.
