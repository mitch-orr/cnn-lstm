import tensorflow as tf
import tflearn
import numpy as np
import cv2
import os
import sys
import time

n_input = 334
n_steps = 596
n_hidden = 128
n_classes = 7
learning_rate = 0.001
training_iters= 10000
batch_size = 64
display_step = 10

annotations = np.loadtxt(fname='tool_video_01.txt', dtype=int, delimiter='\t', skiprows=1)
frames = []

cap = cv2.VideoCapture('tool_video_01.mp4')

ret = True
count = 0
while(ret):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert every 25th frame (as they are annotated)
    if count%25 == 0:
    	f = np.array(frame)
    	#print(f.shape)
    	#tensor = tf.convert_to_tensor(f)
    	frames.append(f)

    count += 1

data = tf.placeholder(tf.int32, [batch_size, n_steps])
target = tf.placeholder(tf.float32, [None, 21])

lstm = tf.contrib.rnn.BasicLSTMCell(n_steps, state_is_tuple=False)

#hidden_state = tf.zeros([batch_size, n_hidden])
#current_state = tf.zeros([batch_size, n_hidden])

#state = hidden_state, current_state

initial_state = state = tf.zeros([batch_size, lstm.state_size])
print(lstm.state_size)
probabilities = []
loss = 0.0

for idx in range(n_steps):
	output, state = lstm(data[:,idx], state)

#Resize/crop etc - multiple for data augmentation?
