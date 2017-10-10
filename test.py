from __future__ import division, print_function, absolute_import

import numpy as np
import tflearn
import tensorflow as tf
import cv2
import sys
import gc

sess = tf.Session()
annotations = np.loadtxt(fname='tool_video_01.txt', dtype=int, delimiter='\t', skiprows=1, usecols=(1,2,3,4,5,6,7))
test_annotations = np.loadtxt(fname='tool_video_03.txt', dtype=int, delimiter='\t', skiprows=1, usecols=(1,2,3,4,5,6,7))

frames = []
test_frames = []

cap = cv2.VideoCapture('tool_video_01.mp4')

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
frames = tf.image.resize_image_with_crop_or_pad(frames, 402, 716)
frames = frames[:,:,:,0]
cap.release()

cap2 = cv2.VideoCapture('tool_video_03.mp4')

ret = True
count = 0
while(ret):
    # Capture frame-by-frame
    ret, frame = cap2.read()
    if frame is None:
        break

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert every 25th frame (as they are annotated)
    if count%25 == 0:
        f = np.array(grey)
        #tensor = tf.convert_to_tensor(f, dtype=tf.uint8)
        test_frames.append(f)

    count += 1

test_frames = np.asarray(test_frames)
cap2.release()
'''shape=[None, 434, 774]'''

with sess.as_default():
    a = frames.eval()
    
    del frames

    #new = np.concatenate((a, test_frames))
    #del a
    #del test_frames

    #new_ann = np.concatenate((annotations, test_annotations))
    #del annotations
    #del test_annotations
    gc.collect()

    net = tflearn.input_data(shape=[None, 402, 716])
    net = tflearn.lstm(net, 128, return_seq=True)
    net = tflearn.lstm(net, 128)
    net = tflearn.fully_connected(net, 7, activation='sigmoid')
    net = tflearn.regression(net, optimizer='adam',
                             loss='binary_crossentropy', name="output1")

    model = tflearn.DNN(net, tensorboard_verbose=2)
    model.fit(test_frames, test_annotations, n_epoch=1, validation_set=0.2, show_metric=True,
    snapshot_step=100)

    model.predict(a[0])

