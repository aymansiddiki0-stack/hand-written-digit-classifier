# Model card — `cnn-20260827-171107` (compact-cnn)

## Architecture
CompactCnn: Conv(1→32,3×3) → ReLU → Conv(32→64,3×3) → ReLU → MaxPool(2) →
Dropout(0.25) → FC(9216→128) → ReLU → Dropout(0.5) → FC(128→10). ~1.2M
parameters. Trained with Adam (lr 1e-3, constant), cross-entropy, batch 256,
seed 20260802, 10 epochs, best epoch 10 by validation accuracy. CPU-only
training (~50-60s/epoch on 1 CPU, one outlier epoch at 205s).

## Training data
MNIST 60k training images (54k train / 6k validation via seeded split
20260802; split indices checksummed). Official 10k test set never used for
tuning.

## Evaluation
MNIST test: **accuracy 0.9917, macro F1 0.9916**
(`artifacts/metrics/cnn-20260827-171107_metrics.json`). Meets the required
≥0.99 accuracy / ≥0.990 macro F1 release gate. All per-class recalls ≥ 0.98;
weakest classes are digit 9 (recall 0.9832) and digit 6 (recall 0.9864).

## Error analysis
83 test errors out of 10,000 samples. No single confusion pair dominates —
the top three are tied at 5 errors each: 9→7, 7→2, 5→3. Digit 9 is the
single most error-prone class overall, contributing to four separate
confusion pairs (9→7, 9→5, 9→8, 9→4; 15 errors total), consistent with its
being the lowest-recall class. No augmentation has been applied to this run;
whether shift augmentation meaningfully reduces these specific confusions
(9 in particular) is worth checking as a follow-up, though the model already
clears the release gate without it.

## Identity
- run_id: `cnn-20260827-171107`
- checkpoint sha256: see `artifacts/run_metadata/cnn-20260827-171107.json`
