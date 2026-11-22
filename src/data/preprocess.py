"""STARD | src/data/preprocess.py

Owner   : Orang 1 (Data & Preproc Lead)
Status  : PLACEHOLDER - structure only, no implementation yet.

Responsibility
--------------
- Video loading and decoding for the raw dataset.
- Face detection + tracking, crop around face, resize to 224x224.
- 16-frame sampling per video (consecutive or uniform).
- Identity-level train/val/test split generation written as .csv.

Contract
--------
- Writes datasets/processed/ frame crops and split index .csv.
- Downstream consumer: src/data/dataset.py.

TODO
----
- [ ] Integrate face detector/tracker and record failure rate
- [ ] Implement consecutive vs uniform sampling switch
- [ ] Emit identity-level split files without identity leakage
"""

# TODO(orang-1): implement preprocessing pipeline described above.
