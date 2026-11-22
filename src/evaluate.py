"""STAF-Net | src/evaluate.py

Owner   : Person 3
Status  : PLACEHOLDER - structure only, no implementation yet.

Responsibility
--------------
- Final evaluation of a trained checkpoint on the held-out test split.
- Produce the standard results table, ROC/confusion plots, and error lists.

Contract
--------
- Usage: python -m src.evaluate --config configs/fusion.yaml \\
           --checkpoint checkpoints/fusion_A8.pt --split test
- Writes results/metrics/<run>.json and results/plots/<run>.png
- Threshold comes from validation; test is evaluated once

TODO
----
- [ ] Load checkpoint into the matching model class
- [ ] Test-split inference + metrics via src/utils/metrics.py
- [ ] ROC / confusion / gate-distribution plots
- [ ] False positive and false negative clip lists for error analysis
- [ ] Append the row to the results table in README
"""

# TODO(person-3): implement the evaluation entry point described above.
