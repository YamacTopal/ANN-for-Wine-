# -*- coding: utf-8 -*-
"""
Created on Sun May 12 14:09:37 2024

@author: Yamaç
"""

from ucimlrepo import fetch_ucirepo 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import pandas as pd
import numpy as np

# fetch dataset 

wine = fetch_ucirepo(id=109) 
  
# data (as pandas dataframes) 
X = wine.data.features 
y = wine.data.targets 


y = pd.get_dummies(y,columns=['class'], dtype="float32")
for i in range(len(y)):
    for j in range(0,3):
        if y.iloc[i,j] == True:
            y.iloc[i,j] = 1
        else:
            y.iloc[i,j] = 0
            


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)





#Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

y_test = np.array(y_test)
y_train = np.array(y_train)

X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        
        self.fc1 = nn.Linear(in_features=X.shape[1], out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=3)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        
        return x

model = ANN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs=100

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train.to(torch.float32))
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1} / {epochs}], Loss: {loss.item():.4f}')
        
        
with torch.no_grad():
    model.eval()
    y_pred = model(X_test.to(torch.float32))
    for i in range(len(y_pred)):
        for j in range(0, 3):
            if y_pred[i,j] >= 0.5:
                y_pred[i, j] = 1
            else:
                y_pred[i, j] = 0
  
    test_accuracy = torch.sum(y_pred == y_test) / y_test.shape[0]
    print(f'Test Accuracy: {test_accuracy.item():.4f}')

predicted = model(X_test.to(torch.float32))
predicted = predicted.detach().numpy()
        
predicted = (predicted > 0.5)
    

    
f1 = f1_score(y_test, predicted, average='macro')
print("F1-score (Macro):", f1)
acc = accuracy_score(y_test, predicted)
print("Accuracy:", acc)