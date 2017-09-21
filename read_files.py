import numpy as np
import tflearn
import tensorflow as tf
import cv2
import sys
import gc

frames = []

for i in range(1,4):

	cap = cv2.VideoCapture('tool_video_0' + str(i) + '.mp4')

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
	    	f = np.array(grey)
	    	#tensor = tf.convert_to_tensor(f, dtype=tf.uint8)
	    	frames.append(f)

	    count += 1
	  
	cap.release()

frames = np.asarray(frames)
gc.collect()
#np.save('vid_data', frames)

print(frames.shape)

'''
cap = cv2.VideoCapture('test.mp4')
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

fc = 0
ret = True

while (fc < frameCount  and ret):
    ret, buf[fc] = cap.read()
    fc += 1
'''