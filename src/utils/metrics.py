"""STAF-Net | src/utils/metrics.py

Owner   : Person 3
Status  : PLACEHOLDER - structure only, no implementation yet.

Responsibility
--------------
- Video-level evaluation for binary real/fake detection.
- AUROC (primary), F1, accuracy, precision, recall, EER.
- Confusion matrix and ROC curve data for results/plots.
- Bootstrap confidence intervals grouped by source_group_id.

Contract
--------
- Pure functions over predictions + labels, no model or dataset dependency.
- Threshold and checkpoint selection use validation data only; test is touched
  once, after selection.
- Outputs are plain dicts / numpy arrays so results/metrics/*.json stays simple.

TODO
----
- [ ] auroc / f1 / accuracy / precision / recall / eer
- [ ] confusion_matrix + roc_curve export
- [ ] group-level bootstrap CI over source_group_id
- [ ] best_threshold_from_val(preds, labels)
- [ ] json exporter for the standard results table (README)
"""

# TODO(person-3): implement metrics described above.
