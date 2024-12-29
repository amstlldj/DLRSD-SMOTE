import collections
from torch.utils.data import TensorDataset
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time
import os
import argparse
import math
import sys
import random
import pandas as pd
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from collections import Counter
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import psutil
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch.nn.init as init


def pcc_similarity(tensor_list_a, tensor_list_b, method="truncate"):
    """
    Calculate the PCC similarity between tensor pairs in the list. The closer to 1, the better

    Parameters:
    - tensor_list_a: tensor list A
    - tensor_list_b: tensor list B
    - method: the way to handle tensors of different lengths, "truncate" means truncation, "pad" means padding

    Return:
    - PCC similarity list
    """
    # Initialize a list to store flattened tensors
    pcc_values = []

    for tensor_a, tensor_b in zip(tensor_list_a, tensor_list_b):
        # Flatten the tensor
        tensor_a = tensor_a.view(-1).cuda()
        tensor_b = tensor_b.view(-1).cuda()
        
        if method == "truncate":
            # Truncate to the length of the shorter tensor
            min_length = min(tensor_a.size(0), tensor_b.size(0))
            tensor_a = tensor_a[:min_length]
            tensor_b = tensor_b[:min_length]
        elif method == "pad":
            # Pad to the length of the longer tensor
            max_length = max(tensor_a.size(0), tensor_b.size(0))
            mean_a = tensor_a.mean() if tensor_a.size(0) > 0 else 0
            mean_b = tensor_b.mean() if tensor_b.size(0) > 0 else 0
            
            tensor_a = torch.nn.functional.pad(tensor_a, (0, max_length - tensor_a.size(0)), value=mean_a.item())
            tensor_b = torch.nn.functional.pad(tensor_b, (0, max_length - tensor_b.size(0)), value=mean_b.item())
        else:
            raise ValueError("The method argument must be 'truncate' or 'pad'")
        
        # Calculate the Pearson correlation coefficient
        pcc = torch.corrcoef(torch.stack([tensor_a, tensor_b]))[0, 1].item()
        pcc_values.append(round(pcc,4))

    return round(np.mean(pcc_values),4)

def js_divergence(tensor_list_a, tensor_list_b, method="truncate"):
    
    js_values = []
    
    for tensor_a, tensor_b in zip(tensor_list_a, tensor_list_b):
        
        tensor_a = tensor_a.view(-1).cuda()
        tensor_b = tensor_b.view(-1).cuda()

        
        if method == "truncate":
            min_length = min(tensor_a.size(0), tensor_b.size(0))
            tensor_a = tensor_a[:min_length]
            tensor_b = tensor_b[:min_length]
        elif method == "pad":
            max_length = max(tensor_a.size(0), tensor_b.size(0))
            mean_a = tensor_a.mean() if tensor_a.size(0) > 0 else 0
            mean_b = tensor_b.mean() if tensor_b.size(0) > 0 else 0
            
            tensor_a = torch.nn.functional.pad(tensor_a, (0, max_length - tensor_a.size(0)), value=mean_a.item())
            tensor_b = torch.nn.functional.pad(tensor_b, (0, max_length - tensor_b.size(0)), value=mean_b.item())
        else:
            raise ValueError("The method argument must be 'truncate' or 'pad'")

        prob_a = torch.softmax(tensor_a, dim=0)  
        prob_b = torch.softmax(tensor_b, dim=0)

        
        m = 0.5 * (prob_a + prob_b)

        js_divergence = 0.5 * (
            torch.sum(prob_a * torch.log(torch.clamp(prob_a / m, min=1e-10))) +
            torch.sum(prob_b * torch.log(torch.clamp(prob_b / m, min=1e-10)))
        )

        js_values.append(js_divergence.item())

    return round(np.nanmean(js_values), 4)


def kl_divergence(tensor_list_a, tensor_list_b, method="truncate"):

    kl_values = []

    for tensor_a, tensor_b in zip(tensor_list_a, tensor_list_b):
        
        tensor_a = tensor_a.view(-1).cuda()
        tensor_b = tensor_b.view(-1).cuda()

        
        if method == "truncate":
            min_length = min(tensor_a.size(0), tensor_b.size(0))
            tensor_a = tensor_a[:min_length]
            tensor_b = tensor_b[:min_length]
        elif method == "pad":
            max_length = max(tensor_a.size(0), tensor_b.size(0))
            mean_a = tensor_a.mean() if tensor_a.size(0) > 0 else 0
            mean_b = tensor_b.mean() if tensor_b.size(0) > 0 else 0
            
            tensor_a = torch.nn.functional.pad(tensor_a, (0, max_length - tensor_a.size(0)), value=mean_a.item())
            tensor_b = torch.nn.functional.pad(tensor_b, (0, max_length - tensor_b.size(0)), value=mean_b.item())
        else:
            raise ValueError("The method argument must be 'truncate' or 'pad'")

        prob_a = torch.softmax(tensor_a, dim=0)  
        prob_b = torch.softmax(tensor_b, dim=0)

        kl_divergence = torch.sum(prob_a * torch.log(torch.clamp(prob_a / (prob_b + 1e-10), min=1e-10)))

        kl_values.append(kl_divergence.item())

    return round(np.nanmean(kl_values), 4)



def cosine_similarity(tensor_list_a, tensor_list_b, method="truncate"):

    cosine_sim_values = []

    for tensor_a, tensor_b in zip(tensor_list_a, tensor_list_b):
        tensor_a = tensor_a.view(-1).cuda()
        tensor_b = tensor_b.view(-1).cuda()
        
        if method == "truncate":
            min_length = min(tensor_a.size(0), tensor_b.size(0))
            tensor_a = tensor_a[:min_length]
            tensor_b = tensor_b[:min_length]
        elif method == "pad":
            max_length = max(tensor_a.size(0), tensor_b.size(0))
            mean_a = tensor_a.mean() if tensor_a.size(0) > 0 else 0
            mean_b = tensor_b.mean() if tensor_b.size(0) > 0 else 0
            
            tensor_a = torch.nn.functional.pad(tensor_a, (0, max_length - tensor_a.size(0)), value=mean_a.item())
            tensor_b = torch.nn.functional.pad(tensor_b, (0, max_length - tensor_b.size(0)), value=mean_b.item())
        else:
            raise ValueError("The method argument must be 'truncate' or 'pad'")
        
        cos_sim = torch.nn.functional.cosine_similarity(tensor_a, tensor_b, dim=0).item()
        cosine_sim_values.append(cos_sim)

    return round(np.mean(cosine_sim_values),4)

def calculate_rmse(tensor_list_a, tensor_list_b, method="truncate"):

    rmse_values = []

    for tensor_a, tensor_b in zip(tensor_list_a, tensor_list_b):
        tensor_a = tensor_a.view(-1).cuda()
        tensor_b = tensor_b.view(-1).cuda()
        
        if method == "truncate":
            min_length = min(tensor_a.size(0), tensor_b.size(0))
            tensor_a = tensor_a[:min_length]
            tensor_b = tensor_b[:min_length]
        elif method == "pad":
            max_length = max(tensor_a.size(0), tensor_b.size(0))
            mean_a = tensor_a.mean() if tensor_a.size(0) > 0 else 0
            mean_b = tensor_b.mean() if tensor_b.size(0) > 0 else 0
            
            tensor_a = torch.nn.functional.pad(tensor_a, (0, max_length - tensor_a.size(0)), value=mean_a.item())
            tensor_b = torch.nn.functional.pad(tensor_b, (0, max_length - tensor_b.size(0)), value=mean_b.item())
        else:
            raise ValueError("The method argument must be 'truncate' or 'pad'")
        
        # 计算余弦相似度
        mse = torch.nn.functional.mse_loss(tensor_a, tensor_b).item()
        rmse = np.sqrt(mse)
        rmse_values.append(rmse)

    return round(np.mean(rmse_values),4)

def mmd(tensor_list_a, tensor_list_b, method="truncate"):
    
    mmd_values = []

    for tensor_a, tensor_b in zip(tensor_list_a, tensor_list_b):
        tensor_a = tensor_a.view(-1).cuda()
        tensor_b = tensor_b.view(-1).cuda()

        if method == "truncate":
            min_length = min(tensor_a.size(0), tensor_b.size(0))
            tensor_a = tensor_a[:min_length]
            tensor_b = tensor_b[:min_length]
        elif method == "pad":
            max_length = max(tensor_a.size(0), tensor_b.size(0))
            mean_a = tensor_a.mean() if tensor_a.size(0) > 0 else 0
            mean_b = tensor_b.mean() if tensor_b.size(0) > 0 else 0
            
            tensor_a = torch.nn.functional.pad(tensor_a, (0, max_length - tensor_a.size(0)), value=mean_a.item())
            tensor_b = torch.nn.functional.pad(tensor_b, (0, max_length - tensor_b.size(0)), value=mean_b.item())
        else:
            raise ValueError("The method argument must be 'truncate' or 'pad'")

        # 计算MMD
        K_aa = torch.mean(tensor_a.unsqueeze(0) * tensor_a.unsqueeze(1))
        K_bb = torch.mean(tensor_b.unsqueeze(0) * tensor_b.unsqueeze(1))
        K_ab = torch.mean(tensor_a.unsqueeze(0) * tensor_b.unsqueeze(1))

        mmd_value = K_aa + K_bb - 2 * K_ab
        mmd_values.append(mmd_value.item())

    return round(np.nanmean(mmd_values), 4)

def emd(tensor_list_a, tensor_list_b, method="truncate"):

    emd_values = []

    for tensor_a, tensor_b in zip(tensor_list_a, tensor_list_b):
        tensor_a = tensor_a.view(-1).cuda()
        tensor_b = tensor_b.view(-1).cuda()

        if method == "truncate":
            min_length = min(tensor_a.size(0), tensor_b.size(0))
            tensor_a = tensor_a[:min_length]
            tensor_b = tensor_b[:min_length]
        elif method == "pad":
            max_length = max(tensor_a.size(0), tensor_b.size(0))
            mean_a = tensor_a.mean() if tensor_a.size(0) > 0 else 0
            mean_b = tensor_b.mean() if tensor_b.size(0) > 0 else 0
            
            tensor_a = torch.nn.functional.pad(tensor_a, (0, max_length - tensor_a.size(0)), value=mean_a.item())
            tensor_b = torch.nn.functional.pad(tensor_b, (0, max_length - tensor_b.size(0)), value=mean_b.item())
        else:
            raise ValueError("The method argument must be 'truncate' or 'pad'")

        emd_value = torch.mean(torch.abs(tensor_a - tensor_b))
        emd_values.append(emd_value.item())

    return round(np.nanmean(emd_values), 4)

def plot_metrics(losses, mean_cs_list, mean_pcc_list, mean_rmse_list,acc_list,mean_js_list,mean_kl_list,mean_mmd_list,mean_emd_list,save_path=None):
    """Draw loss and evaluation metric graphs directly based on the data in the current list"""
    plt.subplot(3, 3, 1)
    plt.plot(losses, label='Loss', color='orange')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend(loc='upper right')

    plt.subplot(3, 3, 2)
    plt.plot(range(len(mean_cs_list)), mean_cs_list, label='Mean CS', color='green')
    plt.xlabel('Iterations')
    plt.ylabel('CS')
    plt.title('Mean CS Over Time')
    plt.legend(loc='lower right')

    plt.subplot(3, 3, 3)
    plt.plot(range(len(mean_pcc_list)), mean_pcc_list, label='Mean PCC', color='purple')
    plt.xlabel('Iterations')
    plt.ylabel('PCC')
    plt.title('Mean PCC Over Time')
    plt.legend(loc='lower right')

    plt.subplot(3, 3, 4)
    plt.plot(range(len(mean_rmse_list)), mean_rmse_list, label='Mean RMSE', color='red')
    plt.xlabel('Iterations')
    plt.ylabel('RMSE')
    plt.title('Mean RMSE Over Time')
    plt.legend(loc='upper right')

    plt.subplot(3, 3, 5)
    plt.plot(range(len(acc_list)), acc_list, label='ACC', color='black')
    plt.xlabel('Iterations')
    plt.ylabel('ACC')
    plt.title('ACC Over Time')
    plt.legend(loc='lower right')

    plt.subplot(3, 3, 6)
    plt.plot(range(len(mean_js_list)), mean_js_list, label='Mean JS', color='pink')
    plt.xlabel('Iterations')
    plt.ylabel('JS')
    plt.title('Mean JS Over Time')
    plt.legend(loc='upper right')

    plt.subplot(3, 3, 7)
    plt.plot(range(len(mean_kl_list)), mean_kl_list, label='Mean KL', color='brown')
    plt.xlabel('Iterations')
    plt.ylabel('KL')
    plt.title('Mean KL Over Time')
    plt.legend(loc='upper right')

    plt.subplot(3, 3, 8)
    plt.plot(range(len(mean_mmd_list)), mean_mmd_list, label='Mean MMD', color='cyan')
    plt.xlabel('Iterations')
    plt.ylabel('MMD')
    plt.title('Mean MMD Over Time')
    plt.legend(loc='upper right')

    plt.subplot(3, 3, 9)
    plt.plot(range(len(mean_emd_list)), mean_emd_list, label='Mean EMD', color='gold')
    plt.xlabel('Iterations')
    plt.ylabel('EMD')
    plt.title('Mean EMD Over Time')
    plt.legend(loc='upper right')

    plt.tight_layout()  
    
    # If a save path is provided, save the image
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
        
    plt.pause(0.1)  # Pause to update the graph
    plt.show()

def one_hot(labels, class_size, device):
    labels = labels.long()  
    targets = torch.zeros(labels.size(0), class_size, device=device)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets