"""Check overlay video file from S3."""
import tempfile, os, cv2
from app.services.s3_client import S3Client

s3 = S3Client()
key = "overlays/f8a1a9bc-b474-4d3b-9bc5-27e8324485e0/myswing_overlay_fs8nbtty.mp4"

tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
tmp.close()
s3._client.download_file(s3._bucket, key, tmp.name)

size = os.path.getsize(tmp.name)
print(f"File size: {size} bytes")

cap = cv2.VideoCapture(tmp.name)
print(f"Opened: {cap.isOpened()}")
print(f"Frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
print(f"Width: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}")
print(f"Height: {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
print(f"Codec: {codec}")
cap.release()
os.unlink(tmp.name)
