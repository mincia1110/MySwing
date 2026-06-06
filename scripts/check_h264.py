import cv2, os
w = cv2.VideoWriter("/tmp/test_h264.mp4", cv2.VideoWriter_fourcc(*"avc1"), 30, (640, 480))
print(f"H264 (avc1) available: {w.isOpened()}")
w.release()

w2 = cv2.VideoWriter("/tmp/test_x264.mp4", cv2.VideoWriter_fourcc(*"X264"), 30, (640, 480))
print(f"X264 available: {w2.isOpened()}")
w2.release()

w3 = cv2.VideoWriter("/tmp/test_h264_2.mp4", cv2.VideoWriter_fourcc(*"H264"), 30, (640, 480))
print(f"H264 available: {w3.isOpened()}")
w3.release()

for f in ["/tmp/test_h264.mp4", "/tmp/test_x264.mp4", "/tmp/test_h264_2.mp4"]:
    if os.path.exists(f):
        os.unlink(f)
