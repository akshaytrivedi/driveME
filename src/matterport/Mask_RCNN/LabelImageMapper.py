
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow.python.framework.ops import convert_to_tensor

import pickle
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import scipy.misc



# In[2]:

missing_images_list = ["road01_cam_6_video_18_image_list_train.txt", "road01_cam_5_video_18_image_list_train.txt"]
train_dataframe = pd.DataFrame([])
for filename in os.listdir('/scratch/at3577/cvpr/train_video_list/'):
    if filename.endswith('.txt') and filename not in missing_images_list:
        dataframe = pd.read_csv(os.path.join('/scratch/at3577/cvpr/train_video_list/', filename), delimiter=r"\s+", header=None)
        dataframe[1] = dataframe.apply(lambda row : row[1].split('\\')[-1], axis=1)
        dataframe[1] = dataframe.apply(lambda row : os.path.join('/scratch/at3577/cvpr/train_color/',row[1]), axis=1)
        
        dataframe[3] = dataframe.apply(lambda row : row[3].split('\\')[-1], axis=1)
        dataframe[3] = dataframe.apply(lambda row : os.path.join('/scratch/at3577/cvpr/train_label/',row[3]), axis=1)
        train_dataframe = train_dataframe.append(dataframe[[1,3]])

train_dataframe.columns =['image','ground_truth']


# In[3]:

image_mapping_dataframe = train_dataframe
image_paths, ground_truth_paths = image_mapping_dataframe.as_matrix()[:,0] ,image_mapping_dataframe.as_matrix()[:,1]

# create dataset
ground_truth_paths = convert_to_tensor(ground_truth_paths, dtype=tf.string)
dataset = tf.data.Dataset.from_tensor_slices(ground_truth_paths)
num_of_ground_truths = len(image_mapping_dataframe)


# In[4]:

def _map_filenames_to_image(ground_truth_filename):
    ground_truth_string = tf.read_file(ground_truth_filename)
    ground_truth_decoded = tf.image.decode_png(ground_truth_string, channels=0, dtype=tf.uint16)
    return ground_truth_decoded


# In[5]:

def _map_image_to_labels(ground_truth):
    labels = tf.div(ground_truth, 1000)
    labels = tf.reshape(labels,[-1])
    labels, _ = tf.unique(labels)
    return labels


# In[6]:

dataset = dataset.map(map_func=_map_filenames_to_image, num_parallel_calls=4)


# In[7]:

dataset = dataset.map(map_func=_map_image_to_labels, num_parallel_calls=4)


# In[8]:

dataset = dataset.prefetch(4)


# In[9]:

# create an reinitializable iterator given the dataset structure
iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
next_batch = iterator.get_next()
iterator_init_op = iterator.make_initializer(dataset)


# In[10]:

label_to_image_map = dict()


# In[ ]:

# Start Tensorflow session
with tf.Session() as sess:
    sess.run(iterator_init_op)
    
    for i in range(num_of_ground_truths):
        try:
            image_filename = image_mapping_dataframe.iloc[i,0]
            ground_truth_filename = image_mapping_dataframe.iloc[i,1]
            ground_truth = sess.run(next_batch)
        except Exception as e:
            print e
            print 'Error for image file= ' + image_filename + ' groundtruth file= ' + ground_truth_filename
        
        for label in ground_truth:
            if label != 0:
                if label not in label_to_image_map:
                    label_to_image_map[label] = []
                label_to_image_map[label].append(image_filename)


# In[ ]:

with open('/scratch/at3577/cvpr/label_to_image_map.pkl', 'wb') as f:
    pickle.dump(label_to_image_map, f)


# In[ ]:




# In[ ]:



