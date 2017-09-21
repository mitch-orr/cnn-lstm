import numpy as np
import cv2
import tflearn

max_width = 0
max_height = 0

for i in range(1,4):

	cap = cv2.VideoCapture('tool_video_0' + str(i) + '.mp4')
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	
	if width > max_width:
		max_width = width

	if height > max_height:
		max_height = height

print(max_width, max_height)