import numpy as np
import tensorflow as tf
from PIL import Image
import glob

frames = []
tgt_height = 300
tgt_width = 300
sess = tf.Session()

for path in glob.glob('frames/train/grey/test1*'):

	img = Image.open(path)
	img = np.array(img)
	#img = tf.convert_to_tensor(img, dtype=tf.uint8)

	frames.append(img)

frames = np.array(frames)
#frames = frames[..., np.newaxis]

# Explore alterative cropping/resizing methods
frames = tf.image.resize_image_with_crop_or_pad(frames, tgt_height, tgt_width) 
#frames = frames[:,:,:,0]

new = frames.eval(session = sess)
