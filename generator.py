# -*- coding: utf-8 -*-
import numpy as np
import cv2
from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
from scipy.misc import imresize


# utility function to convert greyscale images to rgb
def gray2rgb(img):
    new_img = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_img.append(list(img[i][j])*3)
    new_img = np.array(new_img).reshape(img.shape[0], img.shape[1], 3)
    return new_img

def binarylab(labels,dims,n_labels):
    x = np.zeros([dims[0],dims[1],n_labels])
    for i in range(dims[0]):
        for j in range(dims[1]):
            x[i,j,labels[i][j]]=1
    x = x.reshape(dims[0]*dims[1],n_labels)
    return x

# generator that we will use to read the data from the directory
def data_gen_small(img_dir, mask_dir, lists, batch_size, dims, n_labels):
        while True:
            ix = np.random.choice(np.arange(len(lists)), batch_size)
            imgs = []
            labels = []
            for i in ix:
                # images
                original_img = load_img(img_dir + lists.iloc[i, 0]+".jpg")
                resized_img = imresize(original_img, dims+[3])
                array_img = img_to_array(resized_img)/255
                imgs.append(array_img)
                # masks
                original_mask = cv2.imread(mask_dir + lists.iloc[i, 0] + '.png')
                resized_mask = cv2.resize(original_mask, (dims[0], dims[1]))
                array_mask = binarylab(resized_mask[:, :, 0],dims,n_labels)
                labels.append(array_mask)
            imgs = np.array(imgs)
            labels = np.array(labels)
            yield imgs, labels
