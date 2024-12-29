from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pywt
from joblib import dump,load
import gc
import os
import shutil
import random

def normalize(data):
    """"
    Normalized data

    Parameters:
    data (numpy.ndarray) - data to be normalized

    Returns:
    numpy.ndarray - normalized data
    """
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    normalized_data = (data - data_mean) / data_std
    
    return normalized_data


def split_data_with_overlap(data, window_size=512, overlap_ratio=0.5):
    """"
    Cut time series data into overlapping windows
    param data: input time series data
    param window_size: size of each window
    param overlap_ratio: overlap ratio between windows:param overlap_ratio: overlap ratio between windows
    return: list of data after cutting
    step_size:step_size is a parameter used to determine the step size between each cutting window when cutting data. It is calculated based on the window 
    size (window_size) and the overlap ratio (overlap_ratio).
    For example:
    If the window size is 512 and the overlap ratio is 0.5, then the step_size is 256. This means that each window moves back 256 data points and has half the 
    overlap with the previous window.
    If the overlap ratio is 0.75, then the step_size is 128. This means that each window moves back 128 data points and has 3/4 overlap with the previous 
    window.
    """
    step_size = int(window_size * (1 - overlap_ratio)) 
    slices = []

    for start in range(0, len(data) - window_size + 1, step_size):
        slices.append(data[start:start + window_size])
    
    return np.array(slices)


def make_data(data, train_fraction=0.8):
    """
    Randomly extract the data set, use 80% of the data as the training set and 20% of the data as the test set.

    Parameters:
    data (np.ndarray): input data set, shape (2500, 256)
    train_fraction (float): the proportion used for the training set, the default value is 0.8

    Return:
    tuple: training set and test set
    """
    data = np.array(data)
    num_samples = data.shape[0]
    num_train_samples = int(num_samples * train_fraction)
    indices = np.random.permutation(num_samples)
    train_indices = indices[:num_train_samples]
    test_indices = indices[num_train_samples:]
    train_set = data[train_indices]
    test_set = data[test_indices]
    
    return train_set, test_set

def makeTimeFrequencyImage(data, img_path, img_size,sampling_period=1.0/12000,totalscale=128,wavename='cmor1-1'):
    """
    CWT-module
    
    Generate a time-frequency image and save it to the specified path.
    Parameters:
    data (numpy.ndarray): Input data, usually one-dimensional time series data.
    img_path (str): The save path of the generated image, including the file name and format.
    img_size (tuple): The size of the generated image, for example (128, 128).
    """
    fc = pywt.central_frequency(wavename)
    cparam = 2*fc*totalscale
    scales = cparam/np.arange(totalscale,0,-1)
    for i in range(data.shape[0]):
        plt.figure(figsize=img_size)
        coeffs, freqs = pywt.cwt(data[i,:], scales,wavename,sampling_period)
        amp = np.abs(coeffs)
        t = np.linspace(0, sampling_period, 256, endpoint=False)
        c = plt.contourf(t,freqs,amp,cmap='jet')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(img_path+f'time_frequency_image_{i}.png')
        plt.close()
    gc.collect()
    print('over')


def GenerateImageDataset(data_set,path_list,img_size,sampling_period=1.0/12000,totalscale=128,wavename='cmor1-1'):
    """
    data_set: is a data list, which stores training, validation, and test sets
    path_list: is a path list, which stores the storage paths of various types of data in various data sets
    This function will generate various types of images for each data set and store them in the corresponding locations
    """""
    for i in range(len(path_list)):
        user_input = input("Whether to continue execution (0 to end/1 to execute the next image generation):")
        if user_input == '1':
           makeTimeFrequencyImage(data_set[i],path_list[i],img_size,sampling_period,totalscale,wavename)
        else:
            print("over!")
            break
    gc.collect() 
    print('all over')


def copy_images(train_set_dir, target_dir, num_images):
    source_folder_0 = os.path.join(train_set_dir, '0')
    target_folder_0 = os.path.join(target_dir, '0')

    if os.path.exists(source_folder_0):
        os.makedirs(target_folder_0, exist_ok=True)
        images_0 = [f for f in os.listdir(source_folder_0) if os.path.isfile(os.path.join(source_folder_0, f))]
        
        for image in images_0:
            src = os.path.join(source_folder_0, image)
            dst = os.path.join(target_folder_0, image)
            shutil.copy(src, dst)

    for i in range(1, 5):
        source_folder = os.path.join(train_set_dir, str(i))
        target_folder = os.path.join(target_dir, str(i))

        if os.path.exists(source_folder):
            os.makedirs(target_folder, exist_ok=True)
            images = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

            num_to_copy = min(num_images, len(images))

            selected_images = random.sample(images, num_to_copy)

            for image in selected_images:
                src = os.path.join(source_folder, image)
                dst = os.path.join(target_folder, image)
                shutil.copy(src, dst)
"""
This function can randomly extract a specified number of images from the training sets of categories 1-4 to construct training sets with different BRs, while the training set of category 0 keeps the original number
"""


def copy_images(train_set_dir, target_dir, num_images):
    for i in range(5):  
        source_folder = os.path.join(train_set_dir, str(i))
        target_folder = os.path.join(target_dir, str(i))

        if os.path.exists(source_folder):
            os.makedirs(target_folder, exist_ok=True)
            images = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f)) and f.lower().endswith(('.jpg', '.png'))]
            
            num_to_copy = min(num_images, len(images))

            selected_images = random.sample(images, num_to_copy)

            for image in selected_images:
                src = os.path.join(source_folder, image)
                dst = os.path.join(target_folder, image)
                shutil.copy(src, dst)
    print('over!')

def copy_and_modify_images(a_folder, b_folder, x, y):
    if not os.path.exists(b_folder):
        os.makedirs(b_folder)

    for i in range(5):
        a_subfolder = os.path.join(a_folder, str(i))
        b_subfolder = os.path.join(b_folder, str(i))

        if not os.path.exists(a_subfolder):
            continue
        if not os.path.exists(b_subfolder):
            os.makedirs(b_subfolder)

        images_a = [f for f in os.listdir(a_subfolder) if os.path.isfile(os.path.join(a_subfolder, f))]
        images_b = [f for f in os.listdir(b_subfolder) if os.path.isfile(os.path.join(b_subfolder, f))]

        if i == 0:
            to_delete_b = random.sample(images_b, min(y, len(images_b)))
            for img in to_delete_b:
                os.remove(os.path.join(b_subfolder, img))
            to_copy_a = random.sample(images_a, min(y, len(images_a)))
            for img in to_copy_a:
                shutil.copy(os.path.join(a_subfolder, img), b_subfolder)
        else:
            to_delete_b = random.sample(images_b, min(x, len(images_b)))
            for img in to_delete_b:
                os.remove(os.path.join(b_subfolder, img))
            to_copy_a = random.sample(images_a, min(x, len(images_a)))
            for img in to_copy_a:
                shutil.copy(os.path.join(a_subfolder, img), b_subfolder)

