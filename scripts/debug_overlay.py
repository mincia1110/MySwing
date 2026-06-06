"""Debug: simulate what _run_report_generation receives and check pose data."""
import json
from app.db.session import sync_session_factory
from app.db.models import AnalysisResultTable
import uuid

# The issue: report_generator receives pose_result from the pipeline
# Let's check if the pose_sequence is actually populated in the inter-step data

# Simulate: run pose estimation on a single frame to verify it works
import cv2
import numpy as np
from app.pipeline.pose_estimator import PoseEstimator

# Load one frame from the video
from app.services.s3_client import S3Client
import tempfile, os

s3 = S3Client()
tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
tmp.close()
s3._client.download_file(s3._bucket, "uploads/8fb47283-30a0-46c3-8e9a-fea98fc7de24/swing_short.mp4", tmp.name)

cap = cv2.VideoCapture(tmp.name)
ret, frame = cap.read()
cap.release()
os.unlink(tmp.name)

if not ret:
    print("ERROR: Could not read frame")
    exit(1)

print(f"Frame shape: {frame.shape}")

# Run pose estimation on this frame
estimator = PoseEstimator(min_confidence=0.5, static_image_mode=True)
if not estimator.is_available:
    print("ERROR: MediaPipe not available")
    exit(1)

pose = estimator.process_frame(frame, frame_index=0)
estimator.close()

print(f"Keypoints detected: {len(pose.keypoints)}")
print(f"Overall confidence: {pose.overall_confidence:.3f}")
print(f"Is low confidence: {pose.is_low_confidence}")

if pose.keypoints:
    print("\nFirst 5 keypoints:")
    for kp in pose.keypoints[:5]:
        print(f"  {kp.name}: ({kp.x:.3f}, {kp.y:.3f}) conf={kp.confidence:.3f}")
else:
    print("\nNO KEYPOINTS DETECTED - this is why overlay is empty!")
