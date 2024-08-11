import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


dataset = np.loadtxt('diabetes.csv', delimiter=',')
x = dataset[:,0:8]
y = dataset[:,8]

x = torch.tensor(x,dtype=torch.float32)
y = torch.tensor(y,dtype=torch.float32).reshape(-1,1)

model = nn.Sequential(
    nn.Linear(8, 12),
    nn.ReLU(),
    nn.Linear(12, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
    )

print(model)

class pimaclassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1=nn.Linear(8,12)
        self.act1=nn.ReLU()
        self.hidden2=nn.Linear(12,8)
        self.act2=nn.ReLU()
        self.output=nn.Linear(8,1)
        self.outputact=nn.Sigmoid()
    
    def forward(self,x):
        x=self.act1(self.hidden1(x))
        x=self.act2(self.hidden2(x))
        x=self.output(self.outputact(x))
        return x
model= pimaclassifier()

print(model)


loss_fn = nn.BCELoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 100
batch_size = 10

for epoch in range(n_epochs):
    for i in range(0, len(x), batch_size):
        xbatch = x[i:i+batch_size]
        y_pred = model(xbatch)
        ybatch = y[i:i+batch_size]
        loss = loss_fn(y_pred,ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Finished epoch {epoch}, latest loss {loss}')