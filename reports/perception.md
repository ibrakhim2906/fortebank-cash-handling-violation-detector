# V1 perception evidence

The existing pinned Apache-2.0 OpenCV Zoo YOLOX-S weights are reused with their verified RGB preprocessing, hash and source revision. Inference retains people, bags, cup/bottle, chair/table, display/laptop, phone, cabinet proxy and book observations. Anonymous within-recording tracks use IoU; no face or persistent identity model is involved. Sampling is sequential2fps with source decoder/frame times. Cache identity is evaluation-only full SHA, never an inference lookup.

Three annotated source samples are private media in reports/perception_samples/ (ignored by Git). Inspection establishes limitations, not action accuracy:

- A foreground pair is detected in one sample; the safe/cabinet is mislabeled as a display.
- Another sample misses the visible seated person. A missing detector box therefore cannot establish absence.
- A small person behind glass is removed by the height filter, but a nearer client is also rejected. Geometry alone has not solved glass/room separation. Keep occupancy/roles uncertain until scene corroboration; do not fire access or departure rules from these counts alone.

The full cache benchmark will be appended after all40 unique sources complete. These are pretrained object observations, not violation recognition or a trained model improvement. The unchanged numeric fallback replay remains AUC0.5098039215686274 (10/17 recall,10/15 false alarms); expanded observations have not been fitted as that model's features. Its20-shuffle diagnostic mean0.4718627450980392 has no suspicious feature flag. The default tabular train/sanity entrypoints fail on missing raw/train.csv; the correct video adapters pass. No synthetic tabular score substitutes for this case.

Completed all40 source groups: {"videos": 40, "frames": 2620, "total_sec": 749.8603572579887, "worst_sec": 68.88005419100227, "sec_per_frame": 0.28620624322823995}. Benchmark includes concurrent test workload; no recognition accuracy is implied.
