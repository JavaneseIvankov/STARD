"""STAF-Net | src/data/face_tracking.py

Owner   : Person 1 + Person 2 (shared)
Status  : PLACEHOLDER - structure only, no implementation yet.

Responsibility
--------------
- Decode a video (FFmpeg or OpenCV) and sample a ~2 second window of 16
  temporally ordered frames.
- Detect faces (RetinaFace or equivalent), track one face across the window
  with landmark/IoU matching, crop a square region with ~20% context margin.
- Resize crops and cache them on local SSD so training never decodes video.
- Log decode and face-detection failures instead of silently dropping videos.

Contract
--------
- Writes per-clip crops to data/clips/<video_id>/*.jpg plus a clip record with
  clip_start_sec, face box trajectory, and preprocessing_version.
- Same time interval for both branches: DINO takes frames 0,5,10,15 of the
  window; V-JEPA takes all 16.
- Consumers: src/data/video_dataset.py

TODO
----
- [ ] Video decode with explicit failure reporting
- [ ] 16-frame window sampler (deterministic for val/test)
- [ ] RetinaFace detection + tracker with IoU/landmark association
- [ ] Square crop with margin, resize helper for 224 and 384
- [ ] Clip cache writer + resume-safe skipping
"""

# TODO(person-1,person-2): implement face tracking and clip caching described above.
