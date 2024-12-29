import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from PIL import Image
import csv
import time 
import random  
import numpy as np
import zipfile

def set_random_seed(seed):  
    random.seed(seed)  # Python 随机数生成器的种子  
    np.random.seed(seed)  # NumPy 随机数生成器的种子  
    torch.manual_seed(seed)  # PyTorch CPU 的随机数生成器的种子  
    if torch.cuda.is_available():  
        torch.cuda.manual_seed(seed)  # PyTorch GPU 的随机数生成器的种子  
        torch.cuda.manual_seed_all(seed)  # 对所有 GPU 设置种子  
    torch.backends.cudnn.deterministic = True  # 保证每次运行结果一致  
    torch.backends.cudnn.benchmark = False  # 禁用基于输入大小的优化  

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 10),  
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class VGG13(nn.Module):
    def __init__(self):
        super(VGG13, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 10),  
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class VGG11(nn.Module):
    def __init__(self):
        super(VGG11, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 10),  
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)  

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)  
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []

        for label in range(10):  
            folder = os.path.join(root_dir, str(label))
            if os.path.isdir(folder):
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    if os.path.isfile(file_path):  
                        self.data.append(file_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path).convert('L')  
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

class EarlyStopping:
    def __init__(self, patience=5, min_loss=0.01, max_accuracy=0.99, verbose=False):
        self.patience = patience
        self.min_loss = min_loss  
        self.max_accuracy = max_accuracy  
        self.verbose = verbose
        self.best_loss = np.inf
        self.early_stop = False
        self.counter = 0

    def __call__(self, train_loss, train_accuracy):
        if train_loss < self.min_loss or train_accuracy > self.max_accuracy:
            self.early_stop = True
            if self.verbose:
                print("Early stopping triggered due to loss or accuracy threshold.")
            return

        if train_loss < self.best_loss:
            self.best_loss = train_loss
            self.counter = 0 
        else:
            self.counter += 1 
            print(f'Stop early after {patience-self.counter} times！')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered due to patience.")

def plot_loss_acc_time_curve(train_losses, train_accuracies, train_times, plot_dir):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    fig, (ax_loss, ax_acc, ax_time) = plt.subplots(1, 3, figsize=(18, 6))

    ax_loss.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', color='blue', linestyle='-', marker='o')
    ax_loss.set_title('Training Loss Curve')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.grid(True)
    ax_loss.legend()

    ax_acc.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy', color='green', linestyle='-', marker='o')
    ax_acc.set_title('Training Accuracy Curve')
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.grid(True)
    ax_acc.legend()

    ax_time.plot(range(1, len(train_times) + 1), train_times, label='Training Time', color='red', linestyle='-', marker='o')
    ax_time.set_title('Training Time Curve')
    ax_time.set_xlabel('Epoch')
    ax_time.set_ylabel('Time (s)')
    ax_time.grid(True)
    ax_time.legend()

    loss_acc_time_plot_path = f'{plot_dir}/training_loss_accuracy_time_curve.png'
    fig.savefig(loss_acc_time_plot_path)
    print(f"Training loss, accuracy, and time curve saved at {loss_acc_time_plot_path}")

    plt.show()  

def train(model, train_loader, criterion, optimizer, num_epochs=100, early_stopping=None, device='cuda', output_file='results/training_results.csv'):
    model.train()
    
    train_losses = []
    train_accuracies = []
    train_times = []  

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()  
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        
        train_losses.append(epoch_loss)  
        train_accuracies.append(epoch_acc)  
        
        epoch_time = time.time() - epoch_start_time
        train_times.append(epoch_time)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2%}, Time: {epoch_time:.2f}s")
        
        if epoch > 20 and early_stopping is not None:
            early_stopping(epoch_loss, epoch_acc)
            if early_stopping.early_stop:
                print(f'Early stopping at epoch {epoch + 1}')
                break
    
    total_training_time = time.time() - start_time
    print(f"Total training time: {total_training_time:.2f} seconds")
    
    while len(train_losses) < num_epochs:
        train_losses.append(None)
        train_accuracies.append(None)
        train_times.append(None)
    
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Loss', 'Accuracy', 'Training Time (s)'])  
        
        for epoch in range(num_epochs):
            writer.writerow([epoch+1, train_losses[epoch], train_accuracies[epoch], train_times[epoch]])  
    
    return train_losses, train_accuracies, train_times

def plot_confusion_matrix(cm, class_names, save_path=None):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def test_model_with_metrics(model, test_loader, class_names, device, save_path='results/confusion_matrix.png', output_file='results/metrics.txt'):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    cm = confusion_matrix(all_labels, all_predictions)

    plot_confusion_matrix(cm, class_names, save_path)

    accuracy = (np.array(all_predictions) == np.array(all_labels)).mean()
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

    f1 = f1_score(all_labels, all_predictions, average='weighted')
    print(f'F1 Score: {f1:.4f}')

    sensitivity_per_class = []
    specificity_per_class = []

    for i in range(len(class_names)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        sensitivity_per_class.append(sensitivity)
        specificity_per_class.append(specificity)

    g_mean = np.sqrt(np.mean(sensitivity_per_class) * np.mean(specificity_per_class))
    print(f'G-Mean: {g_mean:.4f}')

    with open(output_file, 'a') as f:
        f.write(f'Accuracy: {accuracy * 100:.2f}%\n')
        f.write(f'F1 Score: {f1:.4f}\n')
        f.write(f'G-Mean: {g_mean:.4f}\n')
        f.write('------------------------------\n')

    return accuracy, f1, g_mean

def hook_fn(module, input, output):
    global features
    features = output

def get_vgg16_features_and_tsne(model, loader, device):
    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            features_list.append(outputs.cpu().numpy())  
            labels_list.append(labels.cpu().numpy())      

    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    features_tsne = tsne.fit_transform(features)

    return features_tsne, labels

def plot_tsne(features_tsne, labels, class_names,save_path=None):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.legend(handles=scatter.legend_elements()[0], labels=class_names)
    plt.title('t-SNE Visualization of VGG16 Features')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    if save_path:
       plt.savefig(save_path)  
    plt.show()

def zip_folder(folder_path, zip_name):
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, folder_path))



