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
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import shutil
import os
from joblib import dump,load
import Framework
import MemoryDataset
import Visualization

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

seed = 42  
os.environ['PYTHONHASHSEED'] = str(seed)  
torch.manual_seed(seed)   
np.random.seed(seed)      
random.seed(seed)         
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.makedirs("images", exist_ok=True)
for i in range(10):
    os.makedirs(f"images/{i}", exist_ok=True)
os.makedirs("data", exist_ok=True)

parser = argparse.ArgumentParser() 
parser.add_argument("--n_epochs", type=int, default=1500, help="number of epochs of training") 
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0005, help="adam: learning rate") 
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--gen_size", type=int, default=10, help="the sqrt number of gen size to save")
parser.add_argument("--total_gen_size", type=int, default=20000, help="the total size number of gen size need to save")
parser.add_argument("--visual", type=int, default=1, help="visual print number") 
parser.add_argument("--build_quality", type=int, default=0.6, help="the build_quality to save imgs")
parser.add_argument("--data_load", type=str, default='D:/jupyternotebook/X-1/CWT-IL-ACSAWGAN-GP/Data/CWRU/BR1_400_train_set_balance', help="load data")
parser.add_argument("--n_acc", type=int, default=0.6, help="save acc") # Filtering module setting
parser.add_argument("--n_att", type=int, default=1, help="Alternating training times")
parser.add_argument("--data_save", type=str, default='data', help="save data")
parser.add_argument("--img_save", type=str, default='images', help="save img")
opt = parser.parse_args([])
print(opt)
print(opt.lr)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),  
    transforms.RandomHorizontalFlip(),  
    transforms.ToTensor(),  
    transforms.Normalize((0.5,), (0.5,))  
])
train_dataset = MemoryDataset.MemoryDataset(root=opt.data_load, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

model = Framework.DSEASMOTE().to(device)
gen_loss_function = nn.MSELoss()  
aux_loss_function = nn.CrossEntropyLoss()  
model_optim = torch.optim.Adam(model.parameters(), lr=opt.lr)

# Create the figure and axes
plt.figure(figsize=(10, 5))
plt.ion()
# Initialize loss list and statistics
losses = []
mean_cs_list = []
mean_pcc_list = []
mean_rmse_list = []
mean_js_list = []
mean_kl_list = []
mean_mmd_list = []
mean_emd_list = []
acc_list = []
save_mean_cs_list = []
save_mean_pcc_list = []
save_mean_rmse_list = []
save_mean_js_list = []
save_mean_kl_list = []
save_mean_mmd_list = []
save_mean_emd_list = []
save_acc_list = []
save_cs_total = []
save_pcc_total = []
save_rmse_total = []
save_js_total = []
save_kl_total = []
save_mmd_total = []
save_emd_total = []
save_img_num = 0 
correct_predictions = 0
total_generated_samples = 0
# class_count is a dictionary to store the number of generated images for each category
class_count = {i: 0 for i in range(opt.n_classes)}  # num_classes is the total number of categories
save_img_num = 0  # The file name to use to generate the image

#Training and synthesis part
for epoch in range(opt.n_epochs):
    gen_examples = {i: [] for i in range(10)}
    real_examples = {i: [] for i in range(10)}
    if save_img_num == opt.total_gen_size:
       print(f'High quality generated image data has been collected {save_img_num}! End!')
       Visualization.plot_metrics(losses, mean_cs_list, mean_pcc_list, mean_rmse_list, acc_list,mean_js_list,mean_kl_list,mean_mmd_list,mean_emd_list,save_path=os.path.join(opt.img_save, f'final_results_epoch_{epoch}.png'))
       print(f'save final epoch results pictures!')
       break

    for batch_idx, (images, labs) in enumerate(train_loader):
        model.zero_grad()

        images, labs = images.to(device), labs.to(device)

        x_hat, class_output = model(images)

        # Calculate loss
        gen_loss = gen_loss_function(x_hat, images)  # gen loss
        aux_loss = aux_loss_function(class_output, labs) # AFC-Loss
        loss = gen_loss + aux_loss
        loss.backward()
        
        model_optim.step()

        labs_one_hot = Visualization.one_hot(labs, opt.n_classes, device)  
        pred_labels = torch.argmax(class_output, dim=1)  
        correct_predictions += (pred_labels == labs).sum().item()  
        total_generated_samples += images.size(0)  
        accuracy = correct_predictions / total_generated_samples if total_generated_samples > 0 else 0 # G_acc

        #Store real image data by label distribution
        for r_i in range(len(labs)):
            r_label = labs[r_i].item()  # Use original tags directly
            real_examples[r_label].append(images[r_i])

        #Store generated image data and assign them by label
        for g_i in range(len(class_output)):
            g_label = torch.argmax(class_output[g_i]).item()  # Use argmax to get the predicted label
            gen_examples[g_label].append(x_hat[g_i])
            
        # Periodically output loss and count of generated images
        if batch_idx % opt.visual == 0:
           print(f'Epoch: {epoch+1}/{opt.n_epochs},Batch:{batch_idx}/{len(train_loader)},gen Loss: {gen_loss:.4f}, aux Loss:{aux_loss:.4f},Loss:{loss:.4f},ACC:{accuracy*100:.4f}%')
           gen_count = {key: len(value) for key, value in gen_examples.items()}
           real_count = {key: len(value) for key, value in real_examples.items()}
           print(f"Generate image count: {gen_count},total gen:{sum(gen_count.values())}")
           print(f"True Image Count: {real_count},total real:{sum(real_count.values())}")
           print(f"save img:{save_img_num}")

           pcc_total, cs_total, rmse_total, js_total, kl_total, mmd_total, emd_total = [], [], [], [], [], [], []
           for j in range(10):
               pcc_total.append(Visualization.pcc_similarity(gen_examples[j], real_examples[j], method="pad"))
               cs_total.append(Visualization.cosine_similarity(gen_examples[j], real_examples[j], method="pad"))
               rmse_total.append(Visualization.calculate_rmse(gen_examples[j], real_examples[j], method="pad"))
               js_total.append(Visualization.js_divergence(gen_examples[j],real_examples[j],method="pad"))
               kl_total.append(Visualization.kl_divergence(gen_examples[j],real_examples[j],method="pad"))
               mmd_total.append(Visualization.mmd(gen_examples[j],real_examples[j],method="pad"))
               emd_total.append(Visualization.emd(gen_examples[j],real_examples[j],method="pad"))
           print(f'pcc:{pcc_total}')
           print(f'cs:{cs_total}')
           print(f'rmse:{rmse_total}')
           print(f'js:{js_total}')
           print(f'kl:{kl_total}')
           print(f'mmd:{mmd_total}')
           print(f'emd:{emd_total}')
           mean_cs = np.nanmean(cs_total)
           mean_pcc = np.nanmean(pcc_total)
           mean_rmse = np.nanmean(rmse_total)
           mean_js = np.nanmean(js_total)
           mean_kl = np.nanmean(kl_total)
           mean_mmd = np.nanmean(mmd_total)
           mean_emd = np.nanmean(emd_total)
           print(f'mean cs:{mean_cs},mean pcc:{mean_pcc},mean rmse:{mean_rmse},mean js:{mean_js},mean kl:{mean_kl},mean mmd:{mean_mmd},mean emd:{mean_emd},save img num:{save_img_num}')
           losses.append(loss.item())
           mean_cs_list.append(mean_cs)
           mean_pcc_list.append(mean_pcc)
           mean_rmse_list.append(mean_rmse)
           mean_js_list.append(mean_js)
           mean_kl_list.append(mean_kl)
           mean_mmd_list.append(mean_mmd)
           mean_emd_list.append(mean_emd)
           acc_list.append(accuracy)
           Visualization.plot_metrics(losses, mean_cs_list, mean_pcc_list, mean_rmse_list,acc_list,mean_js_list,mean_kl_list,mean_mmd_list,mean_emd_list)

        pcc_save_total = []
        cs_save_total = []
        rmse_save_total = []
        js_save_total = []
        kl_save_total = []
        mmd_save_total = []
        emd_save_total = []
        for k in range(10): 
            pcc_save_total.append(Visualization.pcc_similarity(gen_examples[k],real_examples[k],method="pad"))
            cs_save_total.append(Visualization.cosine_similarity(gen_examples[k],real_examples[k],method="pad"))
            rmse_save_total.append(Visualization.calculate_rmse(gen_examples[k],real_examples[k],method="pad"))
            js_save_total.append(Visualization.js_divergence(gen_examples[k],real_examples[k],method="pad"))
            kl_save_total.append(Visualization.kl_divergence(gen_examples[k],real_examples[k],method="pad"))
            mmd_save_total.append(Visualization.mmd(gen_examples[k],real_examples[k],method="pad"))
            emd_save_total.append(Visualization.emd(gen_examples[k],real_examples[k],method="pad"))
        m_cs = np.nanmean(cs_save_total)
        m_pcc = np.nanmean(pcc_save_total)
        m_rmse = np.nanmean(rmse_save_total)
        m_js = np.nanmean(js_save_total)
        m_kl = np.nanmean(kl_save_total)
        m_mmd = np.nanmean(mmd_save_total)
        m_emd = np.nanmean(emd_save_total)

        if m_cs > opt.build_quality and save_img_num < opt.total_gen_size and accuracy > opt.n_acc:  # Filtering module
            save_mean_cs_list.append(m_cs)
            save_mean_pcc_list.append(m_pcc)
            save_mean_rmse_list.append(m_rmse)
            save_acc_list.append(accuracy)
            save_mean_js_list.append(m_js)
            save_mean_kl_list.append(m_kl)
            save_mean_mmd_list.append(m_mmd)
            save_mean_emd_list.append(m_emd)
            
            save_cs_total.append(cs_total)
            save_pcc_total.append(pcc_total)
            save_rmse_total.append(rmse_total)
            save_js_total.append(js_total)
            save_kl_total.append(kl_total)
            save_mmd_total.append(mmd_total)
            save_emd_total.append(emd_total)
            
            gen_labels = torch.argmax(class_output, dim=1)
            for gen_label in range(len(x_hat)):  
                g_label = gen_labels[gen_label].item()  
                if class_count[g_label] < 2000:  
                   class_dir = os.path.join(opt.img_save, str(g_label))
                   os.makedirs(class_dir, exist_ok=True)  
                   gen_img_path = os.path.join(class_dir, f'gen_img_{save_img_num}.png')
                   save_image(x_hat[gen_label].cpu(), gen_img_path)  
                   class_count[g_label] += 1  
                   save_img_num += 1  

            print(f'Saving images...,we have save {save_img_num} gen imgs and {len(save_mean_cs_list)} data! The epoch is {epoch}.The batch size is {i}.mean cs:{m_cs},mean pcc:{m_pcc},mean rmse:{m_rmse},mean js:{m_js},mean kl:{m_kl},mean mmd:{m_mmd},mean emd:{m_emd},save acc:{accuracy*100}%')
            print(class_count)

    checkpoint_path = os.path.join(opt.data_save, f'checkpoint.pth')
    torch.save({
        'epoch': epoch,
        'encoder_state_dict': model.state_dict(),
        'model_optim_state_dict': model_optim.state_dict(),
        'loss': loss.item(),
        'accuracy': accuracy,
    }, checkpoint_path)

    if epoch == opt.n_epochs - 1:
       Visualization.plot_metrics(losses, mean_cs_list, mean_pcc_list, mean_rmse_list, acc_list,mean_js_list,mean_kl_list,mean_mmd_list,mean_emd_list,save_path=os.path.join(opt.img_save, f'final_results_epoch_{epoch}.png'))
       print(f'save final epoch results pictures!')

plt.ioff()
plt.show()

mean_cs = np.nanmean(save_mean_cs_list)
mean_pcc = np.nanmean(save_mean_pcc_list)
mean_rmse = np.nanmean(save_mean_rmse_list)
mean_js = np.nanmean(save_mean_js_list)
mean_kl = np.nanmean(save_mean_kl_list)
mean_mmd = np.nanmean(save_mean_mmd_list)
mean_emd = np.nanmean(save_mean_emd_list)
mean_acc = np.nanmean(save_acc_list)
data = {
    'DSEA-SMOTE': [mean_cs],
    'others': [None],  
}
df = pd.DataFrame(data, index=['mean_cs'])
df.loc['mean_pcc'] = [mean_pcc, None]
df.loc['mean_rmse'] = [mean_rmse, None]
df.loc['mean_acc'] = [mean_acc,None]
df.loc['mean_js'] = [mean_js, None]
df.loc['mean_kl'] = [mean_kl, None]
df.loc['mean_mmd'] = [mean_mmd, None]
df.loc['mean_emd'] = [mean_emd, None]
df.to_csv(os.path.join(opt.data_save,'training_results.csv'), index=True)
print("Training results saved to training_results.csv")

dump(save_mean_cs_list,'data/save_mean_cs_list')
dump(save_mean_pcc_list,'data/save_mean_pcc_list')
dump(save_mean_rmse_list,'data/save_mean_rmse_list')
dump(save_acc_list,'data/save_acc_list')
dump(save_mean_js_list,'data/save_mean_js_list')
dump(save_mean_kl_list,'data/save_mean_kl_list')
dump(save_mean_mmd_list,'data/save_mean_mmd_list')
dump(save_mean_emd_list,'data/save_mean_emd_list')
dump(save_cs_total,'data/save_cs_total')
dump(save_pcc_total,'data/save_pcc_total')
dump(save_rmse_total,'data/save_rmse_total')
dump(save_js_total,'data/save_js_total')
dump(save_kl_total,'data/save_kl_total')
dump(save_mmd_total,'data/save_mmd_total')
dump(save_emd_total,'data/save_emd_total')
print('save over')

folder_path = opt.img_save
output_filename = 'gen_img_save'
shutil.make_archive(output_filename, 'zip', folder_path)
print(f'The folder is packed into a compressed file {output_filename}.zip')

folder_path_1 = opt.data_save
output_filename_1 = 'gen_data_save'
shutil.make_archive(output_filename_1, 'zip', folder_path_1)
print(f'The folder is packed into a compressed file {output_filename_1}.zip')