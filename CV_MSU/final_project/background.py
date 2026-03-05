import cv2
import numpy as np

# Открытие видео
video = cv2.VideoCapture("road.mp4")

if not video.isOpened():
    print('Error')
    exit()

num_frames = 100
background = None
frame_count = 0

while frame_count < num_frames:
    ok, frame = video.read()
    if not ok:
        break
    frame_float = frame.astype(np.float32)
    if background is None:
        background = np.zeros_like(frame_float)
    background += frame_float / num_frames
    frame_count += 1

background = background.astype(np.uint8)

cv2.imwrite('background.png', background)


video.release()
cv2.destroyAllWindows()
