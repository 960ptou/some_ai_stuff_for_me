from multiprocessing.sharedctypes import Value
import numpy as np
import cv2
from math import pow, sqrt
from glob import glob
from tqdm import tqdm
from PIL import Image
import os
from testing_model1 import transform
import matplotlib.pyplot as plt

def main(image_files):
    rs, gs, bs = [], [],[]
    for img in tqdm(image_files):
        
        
        image = Image.open(img).convert("RGB")
        z = transform(image)

    
        
        image = cv2.imread(img)
        if image.shape[-1] != 3:
            print(img)
        
        

        try:
            b, g, r = cv2.split(image)
        except ValueError as ve:
            continue
        
        continue
        rs.append(r.mean())
        gs.append(g.mean())
        bs.append(b.mean())
    return 1,1
    
    r_mean = sum(rs)/len(rs)
    g_mean = sum(gs)/len(gs)
    b_mean = sum(bs)/len(bs)

    rs = (np.array(rs) - r_mean) **2
    gs = (np.array(gs) - g_mean) **2
    bs = (np.array(bs) - b_mean) **2

    vr = sum(rs)/ (len(rs) - 1)
    vg = sum(gs)/ (len(gs) - 1)
    vb = sum(bs)/ (len(bs) - 1)

    return [r_mean, g_mean, b_mean], [sqrt(vr), sqrt(vg), sqrt(vb)]


if __name__ == "__main__":
    files = glob("./data/*/*")
    means, stds = main(files)

    print("means :", means)
    print("stds :", stds)