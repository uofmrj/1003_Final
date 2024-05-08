import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import random
import logging
import torch.nn.functional as F
from collections import defaultdict
from sklearn.preprocessing import label_binarize
import time
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc

# Step 1: Set up the dataset
dataset_path = './CUB_200_2011/images'  # Update with the actual path

# Define transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
full_dataset = ImageFolder(root=dataset_path, transform=transform)

# load train
train_path = 'ft_set/support_set.pkl'
with open(train_path, 'rb') as file:
    train_set = pickle.load(file)

# load test
test_path = 'ft_set/test_set.pkl'
with open(test_path, 'rb') as file:
    test_set = pickle.load(file)
    


# Step 2: Define the model
model = resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)  # Adjust for 10 classes

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Step 3: Training the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

def train_model():
    model.train()
    for epoch in range(100):  # Number of epochs
        input_datas = []
        labels = []
        for class_id, imgs in train_set.items():
            input_datas.extend(imgs)
            labels.extend([class_id] * len(imgs))  
        input_datas = torch.stack(input_datas).to(device)  # Convert list of tensors to a single tensor and move to the correct device
        labels = torch.tensor(labels, dtype=torch.long).to(device) 
    
        optimizer.zero_grad()
        outputs = model(input_datas)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
import pickle

def test_model():
    model.eval()
    total_test_samples_num = 0
    predictions = []
    labels = []
    probabilities = []
    
    print('============================================================')

    with torch.no_grad():  # Disable gradient computation for inference
        for class_id, this_class_data in test_set.items():
            this_class_samples_num = len(this_class_data)
            total_test_samples_num += this_class_samples_num

            # Convert data list to tensor and transfer it to the correct device
            this_class_data = torch.stack(this_class_data).to(device)

            # Get predictions
            predict_this_class = model(this_class_data)
            predict_probs = F.softmax(predict_this_class, dim=1)
            predict_label = torch.argmax(predict_probs, dim=1).cpu().numpy()  # Move tensor to CPU before converting to numpy
            probabilities.extend(predict_probs.cpu().numpy())  # Store probabilities
            
            # Collect predictions and labels for metrics computation
            predictions.extend(predict_label)
            labels.extend([class_id] * this_class_samples_num)

    # Calculate and print classification metrics
    report = classification_report(labels, predictions, digits=2)
    print(report)
    
    # Prepare for AUC calculation
    classes = np.unique(labels)
    labels_binarized = label_binarize(labels, classes=classes)
    
    # Calculate AUC
    auc_score = roc_auc_score(labels_binarized, probabilities, multi_class='ovr', average='macro')
    
    print(f"Macro Averaged AUC Score: {auc_score:.2f}")

    # ROC metrics
    fpr, tpr, _ = roc_curve(labels_binarized.ravel(), np.array(probabilities).ravel())
    mean_auc = auc(fpr, tpr)

    # Save ROC data
    roc_data = {'mean_fpr': fpr, 'mean_tpr': tpr, 'mean_auc': mean_auc}
    with open('base_roc.pkl', 'wb') as f:
        pickle.dump(roc_data, f)

    return fpr, tpr, mean_auc  # Optional return



start_time = time.time()
train_model()
end_time = time.time()
print(f"Training time is {end_time - start_time}")
test_model()