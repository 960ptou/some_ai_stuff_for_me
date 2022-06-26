from cgi import test
from pkg_resources import working_set
import torch
import torch.nn as nn

from glob import glob

from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch.nn.functional import one_hot

from models.test_cnn_1 import TestNet1
from util import *
from testing_model1_data import TestingModel1DataSet, collate_fn


device = get_available_device()
cat_dog_mean = (0.48820977, 0.45513667, 0.41686828)
cat_dog_std  = (0.11603589, 0.11061857, 0.11545174)

mean = torch.tensor(cat_dog_mean).to(device).view(3,1,1)
std  = torch.tensor(cat_dog_std).to(device).view(3,1,1)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((50,50)),
    transforms.Normalize(cat_dog_mean, cat_dog_std),
])

data_dir = "/Users/weijiechen/Desktop/Personal/Personal_Python_files/IntroToAI/PetImages"
batch_size = 16
epochs = 1

test1_dataset = TestingModel1DataSet(data_dir=data_dir, transform=transform)
train_size = int(len(test1_dataset) * 0.8)
test_size = len(test1_dataset) - train_size
train_dataset, test_dataset = random_split(test1_dataset, [train_size, test_size])


dataloader_train = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
dataloader_test = DataLoader(test_dataset, batch_size=batch_size,shuffle=False)

net = TestNet1()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr = 0.002)
lambda_fun = lambda epoch : 1/(1 + 2.71828 ** epoch)
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda_fun)

if __name__ == "__main__":


    for epoch in range(1, epochs + 1):
        for i, (X,y) in enumerate(dataloader_train):
            net.zero_grad()
            output = net(X)
            y = one_hot(y,2).float()
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            
        if epoch % 5 == 0:
            lr_scheduler.step()
            
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (test_X,test_y) in enumerate(dataloader_test):
                result = net(test_X)
                predicted_class = torch.argmax(result, dim=1)
                
                
                correct += int(sum(torch.where(predicted_class == test_y, 1 , 0)))
                total += batch_size
            print("Accuracy", round(correct/total, 5))