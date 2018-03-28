import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import h5py
from mpl_toolkits.axes_grid1 import ImageGrid
from os import listdir, makedirs
from os.path import join, exists, expanduser
from sklearn.metrics import log_loss, accuracy_score
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input, decode_predictions


def preprocess_data(labels, no_gpu = True):
    INPUT_SIZE = 224
    NUM_CLASSES = 16

    # group by dataframe.groupba()
    labels_group = labels.groupby('breed')
    # labels_group_list = [x for x in labels_group]  # 120 classes
    selected_breed_list = labels_group.count()  # count every classes numbers
    if no_gpu:
        selected_breed_list = selected_breed_list.sort_values(by='id', ascending=False).head(NUM_CLASSES).index #
    else:
        selected_breed_list = selected_breed_list.sort_values(by='id', ascending=False).index
    selected_breed_list = list(selected_breed_list)

    labels = labels[labels['breed'].isin(selected_breed_list)]
    labels['target'] = 1
    labels['rank'] = labels_group.rank()['id']
    # print(labels.head(30))

    # id为行，breed为列，用target的值填充
    labels_pivot = labels.pivot('id', 'breed', 'target')
    labels_pivot = labels_pivot.reset_index().fillna(0)  # 索引

    rnd = np.random.random(len(labels))
    train_idx = rnd < 0.8
    valid_idx = rnd >= 0.8
    y_labels = labels_pivot[selected_breed_list].values
    y_train = y_labels[train_idx]
    y_val = y_labels[valid_idx]
    return y_train, y_val


def read_img(img_id, spilt, size):
    """Read and resize image.
    # Arguments
        img_id: string
        spilt: string 'train' or 'test'.
        size: resize the original image.
    # Returns
        Image as numpy array.
    """
    img = image.load_img(os.path.join(base_dir, spilt, "%s.jpg"%img_id), target_size=size)
    img = image.img_to_array(img)
    return img    # numpy

base_dir = "./"
train_id = os.listdir(os.path.join(base_dir, 'train')) # list
labels = pd.read_csv(os.path.join(base_dir, 'labels.csv'))  # DataFrame
test_id = os.listdir(os.path.join(base_dir, 'test'))
sample_submission = pd.read_csv(os.path.join(base_dir, 'sample_submission.csv'))

# print(len(train_id), labels.shape)   #10222 (10222,2)
# print(len(test_id), sample_submission.shape) # 10357 (10357, 121)
model = ResNet50(weights='imagenet')
# config = model.get_config()
# weights = model.get_weights()

y_train, y_val = preprocess_data(labels, False)