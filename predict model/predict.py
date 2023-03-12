import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import Net


device='cuda'
model=Net().to(device)

new_data_tc= ['H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Sc', 'Ti', 'V', 
            'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 
            'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 
            'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 
            'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu',]

def bracket_hash(formula):
    element = ""
    element_hash = {}
    for x in formula:
        if x.isupper():
            element = x
            element_count=''
        elif x.islower():
            element += x            
        else: 
            element_count += x
            element_hash[element] = element_count
    return element_hash


def formula2vec(formula):
    com=bracket_hash(str(formula))
    array={m:0 for m in new_data_tc}
    index=[]
    for p in com:            
        if p in array:
            array[p]=com[p]
        else:
            array=[]
            print('=====')
    for q in array:
        index.append(array[q])
    return index



model=torch.load(r'ckp\model.pth')
print('model loaded success')


def predict(formula):
    data=formula2vec(formula)
    data=np.array(data).astype(float)
    data=data.reshape(-1,1,86)
    data=torch.tensor(data)
    data = data.to(device).float()
    output = model(data)
    return output[0].item()

formula='O8.44Al0.16Ca2.11Cu3.12Ba1.99Hg0.90'

Tc=predict()
print(Tc)
