# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics


class MIMIC3(torch.utils.data.Dataset):
  def __init__(self, root):
    '''

    :param root: DataFrame (X and Y)
    Need to be Implemented...
    '''
  def __getitem__(self, index):
    return self.X[index], self.y[index]
  def __len__(self):
    return self.length

# Loading the Dataset into DataLoader
train_dataset = MIMIC3()
test_dataset = MIMIC3()
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=32, 
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=32, 
                                           shuffle=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
hidden_size = 300
learning_rate = 0.001

class FFNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FFNet, self).__init__()
        self.net = nn.Sequential(
          nn.Linear(input_size, hidden_size),
          nn.BatchNorm1d(hidden_size),
          nn.ReLU(),
        
          nn.Linear(hidden_size, hidden_size),
          nn.BatchNorm1d(hidden_size),
          nn.ReLU(),
          
          nn.Linear(hidden_size, hidden_size),
          nn.BatchNorm1d(hidden_size),
          nn.ReLU(),
        
          nn.Linear(hidden_size, num_classes)
        )
    def forward(self, x):
        out = self.net(x)
        return out

# Train the model
def train_ffnet(model, train_loader, num_epochs):
  total_step = len(train_loader)
  for epoch in range(num_epochs):
    for i, (data, labels) in enumerate(train_loader):
      data = data.to(device)
      labels = labels.to(device)
      
      # Forward pass
      outputs = model(data)
      loss = criterion(outputs, labels.long())
      
      # Backward and optimize
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      # Display the progress
      if (i+1) % 300 == 0:
        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
               .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        
# Test the model
def test_ffnet(model, test_loader):
  preds = []
  acts = []
  # In test phase, we don't need to compute gradients (for memory efficiency)
  with torch.no_grad():
    for images, labels in test_loader:
      images = images.to(device)
      labels = labels.to(device).long()
      outputs = model(images)
      predicted = outputs.data
      preds.extend(predicted.tolist())
      acts.extend(labels.tolist())

    preds = np.array(preds); acts = np.array(acts)
    r2 = sklearn.metrics.r2_score(acts,preds)

    # Display the result
    print("R2_Score : {}".format(r2))


input_size = 35 # X length (Must be changed)
num_classes = 16 # Y length (Must be changed)
model = (FFNet(input_size, hidden_size, num_classes).to(device)).double()

# Set the loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

train_ffnet(model, train_loader, num_epochs=100)
test_ffnet(model, test_loader)
