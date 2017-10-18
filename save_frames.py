import cv2
import numpy as np
from PIL import Image
import sys

cap = cv2.VideoCapture('tool_video_04.mp4')
length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(length/25)
sys.exit()
frame_no = 0

ret = True
count = 0
while(ret):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if frame is None:
    	break

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert every 25th frame (as they are annotated)
    if count%25 == 0:

        cv2.imwrite('frames/vid_15_frame_' + str(frame_no) + '.jpg', grey)
        print('Saved frame ' + str(frame_no) + ' of vid 15')
        frame_no += 1

    count += 1

cap.release()

print(length, length/25)
# https://www.learnopencv.com/color-spaces-in-opencv-cpp-python/