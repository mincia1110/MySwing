# Pose backend benchmark

This benchmark compares MediaPipe and RTMPose on the same normalized frames and
the same downstream tracking, bat, and phase pipeline. It was run on an Apple M4
Mac with Python 3.11.15, MediaPipe 0.10.21, RTMLib 0.0.15, ONNX Runtime 1.27.0,
OpenCV 4.11.0, and NumPy 1.26.4.

The coverage, confidence, and segment-length measurements below are diagnostic
proxies, not ground-truth keypoint localization error. Impact error is compared
against manually annotated normalized-frame indices.

## Inputs

| Clip | SHA-256 | Frames | Size | Impact frame |
|---|---|---:|---:|---:|
| Indoor side view | `8d3371d9089ba368f346c246cc6ec6451f5cd9ff635e1cdc851ff0582b33ca99` | 180 | 960x720 | 123 |
| Broadcast portrait | `d5c16223fb69f00cf1def5dc5b6351be70d951e9001db8d0352d6bba129d9b1f` | 240 | 606x1080 | 169 |

Both clips were evaluated at 30 FPS with temporal conditioning disabled. RTMPose
ran through ONNX Runtime CPU. Its top-down detector was restricted to one
largest central, temporally continuous person so spectators did not trigger pose
inference.

## Results

### Indoor side view

| Candidate | Pose time | Low-confidence frames | Wrist coverage | Elbow coverage | Median segment CV | Impact error |
|---|---:|---:|---:|---:|---:|---:|
| MediaPipe Full | 3.96s | 38.3% | 77.5% | 83.6% | 0.1989 | abstained |
| MediaPipe Heavy | 10.87s | 49.4% | 64.7% | 78.3% | 0.1246 | abstained |
| RTMPose-s | 7.51s | 14.4% | 93.6% | 93.1% | 0.1956 | abstained |
| RTMPose-m | 36.69s | 13.9% | 93.3% | 93.3% | 0.1832 | abstained |

No candidate had detector-observed bat evidence on this clip, so abstention was
the correct pipeline behavior; no artificial numeric contact error is assigned.

### Broadcast portrait

| Candidate | Pose time | Low-confidence frames | Wrist coverage | Elbow coverage | Median segment CV | Observed bat-line coverage | Impact error |
|---|---:|---:|---:|---:|---:|---:|---:|
| MediaPipe Full | 5.16s | 31.2% | 84.2% | 84.2% | 0.1365 | 0.0% | abstained |
| MediaPipe Heavy | 14.26s | 87.1% | 59.2% | 61.9% | 0.1350 | 31.2% | 15 frames |
| RTMPose-s | 9.91s | 8.3% | 96.7% | 95.8% | 0.1315 | 47.9% | 34 frames |
| RTMPose-m | 46.87s | 0.0% | 100.0% | 100.0% | 0.1216 | 44.2% | 15 frames |

## Decision

RTMPose-m (`balanced`) is the default accuracy-first backend. Across all 420
frames, its wrist coverage was about 97.1% versus 81.3% for MediaPipe Full, and
its low-confidence frame rate was about 6.0% versus 34.3%. It also produced the
best segment-length stability on the broadcast clip and matched MediaPipe Heavy's
15-frame impact error while retaining substantially more wrist and elbow data.

RTMPose-s remains a useful speed-oriented option. RTMPose-x was rejected for the
Apple CPU default: it took 127.4 seconds for the 180-frame side clip, showed worse
segment stability than RTMPose-m, and offered no evidence sufficient to justify
the additional detector/model cost.

Run the comparison again with:

```sh
.venv/bin/python scripts/swing_accuracy_benchmark.py \
  --video /path/to/swing.mp4 \
  --batting-direction right \
  --impact-frame FRAME \
  --candidate static_full \
  --candidate static_heavy \
  --candidate rtmpose_lightweight \
  --candidate rtmpose_balanced \
  --conditioning off \
  --output /tmp/myswing-pose-backends.json
```
