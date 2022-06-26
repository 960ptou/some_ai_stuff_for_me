from torch.utils.data import Dataset
from torch.nn.functional import one_hot
from glob import glob
import os
from PIL import Image

# Let's do binary, (cat & dog) => will top at >= 85%
class TestingModel1DataSet(Dataset):
    def __init__(self, data_dir, transform):
        """
        |---- data_dir
        |-------- a
        |-------- b
        """
        self.images = []
        self.labels = []
        self.names = []
        for i, class_dir in enumerate(sorted(glob(os.path.join(data_dir,"*")))):
            images = sorted(glob(os.path.join(class_dir, "*.jpg")))
            self.images += images
            self.labels += ([i] * len(images))
            self.names  += [os.path.relpath(img, data_dir) for img in images]
            
        self.transform = transform
        
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        return self.transform(img), self.labels[index]
        
    def __len__(self):
        return len(self.images)
    

def collate_fn(batch):
    imgs = []
    ids = []
    for img, id in batch:
        imgs.append(img)
        ids.append(one_hot(id, 2))
    return imgs, ids
        
        