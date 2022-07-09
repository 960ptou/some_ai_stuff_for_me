from cgi import test
from numpy import dtype
from pkg_resources import working_set
import torch
import torch.nn as nn

from glob import glob

from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch.nn.functional import one_hot

from models.models.test_net2 import TestNet2
from util import *
from testing_model1_data import TestingModel1DataSet, collate_fn
import os
from time import time
from tqdm import tqdm
from datetime import datetime
import shutil
import matplotlib.pyplot as plt
import math

device = get_available_device()
print("Device :",device)
cat_dog_mean = (0.48820977, 0.45513667, 0.41686828)
cat_dog_std  = (0.11603589, 0.11061857, 0.11545174)

mean = torch.tensor(cat_dog_mean).to(device).view(3,1,1)
std  = torch.tensor(cat_dog_std).to(device).view(3,1,1)

transform = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.ToTensor(),
    transforms.Normalize(cat_dog_mean, cat_dog_std),
])

data_dir = os.path.join(".","data")
batch_size = 2
epochs = 20000

test1_dataset = TestingModel1DataSet(data_dir=data_dir, transform=transform)
train_size = int(len(test1_dataset) * 0.80)
test_size = len(test1_dataset) - train_size
train_dataset, test_dataset = random_split(test1_dataset, [train_size, test_size])


dataloader_train = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
dataloader_test = DataLoader(test_dataset, batch_size=batch_size,shuffle=False)

net = TestNet2().to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr = 0.2)
#lambda_fun = lambda epoch : 1/( 10 ** epoch)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.9)

model_storage_dir = os.path.join(".","models","checkpoints",str(datetime.now().strftime("%m-%d-%Y-%H-%M-%S")))
os.mkdir(model_storage_dir)
shutil.copy2(os.path.join(".","models","models","test_net2.py"), os.path.join(model_storage_dir, "model.py"))


if __name__ == "__main__":
    for sample in dataloader_train:
        X,y = sample
        break
    #X, y = next(iter(dataloader_train))
    X, y = X.to(device), y.to(device).float()
    start = time()
    print("Training Start")
    avg_loss = []

    net.train()
    for i in tqdm(range(epochs)):
        output = net(X)
        loss = loss_fn(output, y.reshape(-1,1))

        loss.backward()
        optimizer.step()
        if i % 300 == 0:
            print(loss)
            print("Learn Rate:", optimizer.param_groups[0]["lr"])
            lr_scheduler.step()
    """
    for epoch in range(1, epochs + 1):
        print("Current Epoch:",epoch)
        print("Learn Rate:", optimizer.param_groups[0]["lr"])
        net.train()
        for i, (X,y) in tqdm(enumerate(dataloader_train)):
            net.zero_grad()
            output = net(X.to(device))
            y = y.to(device).float()
            loss = loss_fn(output, y.reshape(-1,1))
            loss.backward()
            optimizer.step()
    """
            
        #torch.save(net.state_dict(), os.path.join(model_storage_dir,f"{epoch*len(test1_dataset)}.pth"))
        #lr_scheduler.step(avg_loss[-1])
            

    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (test_X,test_y) in enumerate(dataloader_test):
            result = net(test_X.to(device))
            predicted_class = torch.argmax(result, dim=1)
            correct += int(sum(torch.where(predicted_class == test_y.to(device), 1 , 0)))
            total += batch_size
        print("Accuracy", round(correct/total, 5))

    end = time()
    print(round( (end - start)/60, 3), "Minutes")
    #plt.plot(avg_loss)
    #plt.show()