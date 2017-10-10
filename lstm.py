from __future__ import division, print_function, absolute_import

import numpy as np
import tflearn
import tensorflow as tf
import cv2
import sys

sess = tf.Session()

vid_frames = []
annotations = []

# Alternatively, define file name string ('tool_video_0X'), then add file extension when opening cap/annotations

with sess.as_default():

    for i in range(1,10):

        cap = cv2.VideoCapture('tool_video_0' + i + '.mp4')
        anno = np.loadtxt(fname='tool_video_0' + i + '.txt', dtype=int, delimiter='\t', skiprows=1, usecols=(1,2,3,4,5,6,7))

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

        frames = np.asarray(frames)

        frames = frames[...,None]

        # Explore alterative cropping/resizing methods
        frames = tf.image.resize_image_with_crop_or_pad(frames, 434, 774) # Determine max heaight/with and change
        frames = frames[:,:,:,0]

        cap.release()

        frames = frames.eval()

        vid_frames.append(frames)    
        annotations.append(anno)
    
    net = tflearn.input_data(shape=[None, 434, 774])
    net = tflearn.lstm(net, 128, return_seq=True)
    net = tflearn.lstm(net, 128)
    net = tflearn.fully_connected(net, 7, activation='sigmoid')
    net = tflearn.regression(net, optimizer='adam', loss='binary_crossentropy', name="output1")

    model = tflearn.DNN(net, tensorboard_verbose=2)

    model.fit(vid_frames, annotations, n_epoch=1, validation_set=0.2, show_metric=True,
    snapshot_step=100)

#https://github.com/fchollet/keras/issues/5527
#https://link.springer.com/article/10.1007/s11633-016-1006-2