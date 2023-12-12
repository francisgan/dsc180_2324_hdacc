import pandas as pd
import torch.profiler
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torchvision
from typing import Optional, Union
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

train_set = []
test_set = []

for i in range(1,2):
    data = []
    path = 'data/subject10'+str(i)+'.dat'
    with open(path, 'r') as f:
        d = f.readlines()
        for i in d:
            k = i.rstrip().split(" ")
            data.append([float(i) for i in k]) 

    train_set.append(data)

for i in range(9,10):
    data = []
    path = 'data/subject10'+str(i)+'.dat'
    with open(path, 'r') as f:
        d = f.readlines()
        for i in d:
            k = i.rstrip().split(" ")
            data.append([float(i) for i in k]) 

    test_set.append(data)
 
for k in range(0,54):
    latest_value = 0
    for i in train_set[0]:
        if not np.isnan(i[k]):
            latest_value = i[k]
        else:
            i[k] = latest_value

for k in range(0,54):
    latest_value = 0
    for i in test_set[0]:
        if not np.isnan(i[k]):
            latest_value = i[k]
        else:
            i[k] = latest_value

train_data = []
train_label = []
test_data = []
test_label = []

train_num_sub = len(train_set[0])//1000
for i in range(0,train_num_sub):
    sub = train_set[0][i*1000:i*1000+1000]
    sub_feature = []
    sub_label = []
    for j in sub:
        sub_label.append(j[1])
        j = j[:1] + j[2:]
        sub_feature.append(j)
    train_data.append(sub_feature)
    train_label.append(sub_label)

test_num_sub = len(test_set[0])//1000
for i in range(0,test_num_sub):
    sub = test_set[0][i*1000:i*1000+1000]
    sub_feature = []
    sub_label = []
    for j in sub:
        sub_label.append(j[1])
        j = j[:1] + j[2:]
        sub_feature.append(j)
    test_data.append(sub_feature)
    test_label.append(sub_label)

class DNN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.act1 = nn.ReLU()
        self.dropout = nn.Dropout()

        self.fc2 = nn.Linear(128, 64)
        self.act2 = nn.ReLU()

        self.fc3 = nn.Linear(64, output_dim)
        self.act3 = nn.Softmax(dim=-1)

    def forward(self, x):
        #b, t, c = x.size()
        #x = x.reshape(b, -1)
        out = self.dropout(self.act1(self.fc1(x)))
        out = self.act2(self.fc2(out))
        out = self.act3(self.fc3(out))

        return out
    
def train(model, train_loader, optimizer, criterion, device, num_classes, num_time_points):
    model.train()
    total_loss = 0

    for data, target in train_loader:
        target = target.view(-1)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = output.view(-1, num_classes)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    return average_loss

def test(model, test_loader, criterion, device, num_classes, num_time_points):
    model.eval()
    total_loss = 0
    correct = 0
    predictions = []
    labels = []

    with torch.no_grad():
        for data, target in test_loader:
            target = target.view(-1)
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = output.view(-1, num_classes)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            predictions.extend(pred.view(-1).tolist())
            labels.extend(target.tolist())

    average_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / (len(test_loader.dataset) * num_time_points)

    return average_loss, accuracy, predictions, labels



epochs = 10
learning_rate = 5
num_classes = 25          # Number of output classes per time point
num_time_points = 1000       # Number of time points per subject
num_features = 53           # Number of features per time point
input_dim = 53
output_dim = 25

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DNN(input_dim, output_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

    # Parameters
num_subjects_train = 376   # Number of subjects in the training set
num_subjects_test = 8    # Number of subjects in the test set
num_time_points = 1000       # Number of time points per subject
num_features = 53           # Number of features at each time point
num_classes = 25           # Number of classes for each time point

train_data_tensor = torch.FloatTensor(train_data)  # Shape: [376, 1000, 53]
train_labels_tensor = torch.LongTensor(train_label)  # Shape: [376, 1000]
test_data_tensor = torch.FloatTensor(test_data)  # Shape: [8, 1000, 53]
test_labels_tensor = torch.LongTensor(test_label)  # Shape: [8, 1000]

# Reshaping data and labels for compatibility with the model
train_data_tensor = train_data_tensor.view(-1, 53)  # New shape: [376000, 53]
train_labels_tensor = train_labels_tensor.view(-1)  # New shape: [376000]
test_data_tensor = test_data_tensor.view(-1, 53)  # New shape: [8000, 53]
test_labels_tensor = test_labels_tensor.view(-1)  # New shape: [8000]   

# Creating TensorDatasets
train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
test_dataset = TensorDataset(test_data_tensor, test_labels_tensor)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    # Assuming train_loader and test_loader are defined as per earlier instructions

for epoch in range(epochs):
    train_loss = train(model, train_loader, optimizer, criterion, device, num_classes, num_time_points)
    test_loss, test_accuracy, test_predictions, test_labels = test(model, test_loader, criterion, device, num_classes, num_time_points)

    print(f'Epoch {epoch + 1}:')
    print(f'Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    print("Predictions:", test_predictions)
    print("Actual Labels:", test_labels)

# Parameters
num_time_points_new = 1000   # Number of time points for the new subject
num_features = 53             # This should match the training data

# Random data for demonstration (replace this with actual data)
new_data = torch.randn(1, num_time_points_new, num_features)  # Shape: [1, 1000, 3]

# If your model expects a different input size, you may need to reshape or process the data accordingly

# Ensure the model is in evaluation mode
model.eval()

# Move data to the same device as the model
new_data = new_data.to(device)

# Making predictions
with torch.no_grad():
    output = model(new_data)
    output = output.view(-1, num_classes)
    predictions = output.argmax(dim=1).cpu().numpy()
