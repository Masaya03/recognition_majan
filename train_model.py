import tensorflow as tf
from tensorflow.python.framework.ops import Tensor
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import shutil
import math
from IPython.display import Image as IPImage
from IPython.display import display_jpeg

from model import train_model

print(tf.__version__)

train_dir = 'target_datasets/train'
val_dir = 'target_datasets/val'

backup_dir = './label'
model_dir = "./model"

labels = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
labels.sort()

if os.path.exists(backup_dir):
    shutil.rmtree(backup_dir)

os.makedirs(backup_dir)

with open(backup_dir + '/labels.txt','w') as f:
    for label in labels:
        f.write(label+"\n")

NUM_CLASSES = len(labels)
print("class number=" + str(NUM_CLASSES))

labels = []
with open(backup_dir + '/labels.txt','r') as f:
    for line in f:
        labels.append(line.rstrip())
print(labels)


es = EarlyStopping(monitor="val_loss", patience=30, verbose=0, mode="auto")
tb = TensorBoard(log_dir="./log/tensorlog", histogram_freq=1)

model = train_model(epochs=200,batch_size=220)
model.set_model()
model.complie()
model.fit([es,tb])

# %load_ext tensorboard
# %tensorboard --logdir ./log/tensorlog