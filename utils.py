#!/usr/bin/env python
# coding: utf-8
# %%
import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import random
from tqdm import tqdm
import random

class Data:
    def __init__(self, split, fold_idx, dataset,path):
        
        assert split in ['train', 'test']
        assert fold_idx < 10
        
        self.dataset_root = path
        self.dataset_path = os.path.join(self.dataset_root, dataset)
        self.fold_idx = fold_idx        
        self.split = split
        self.x = None
        self.y = None
        self.load()

    def load(self):
        file = np.load(self.dataset_path, allow_pickle=True)
        X = file['X']
        y = file['y']
        if self.split == 'train':
            fold_split =  file['folds'][self.fold_idx][0]
            
        else:
            fold_split =  file['folds'][self.fold_idx][1]
            
        self.x = X[fold_split]
        self.y = y[fold_split].argmax(1)
        lista_2=[]
        for i in range(len(self.x)):
            x_1 = self.x[i]
            x_1=np.concatenate(x_1, axis=0)
            list_1=[]
            for j in range(50):
                x_2=x_1[j]
                x_3 = x_2[:3] 
                list_1.append(x_3)
            lista_2.append(list_1)

        self.x = np.array(lista_2)
        return self.x, self.y




class DataCat:
    def __init__(self, split, fold_idx, category, dataset,path):
        
        assert split in ['train', 'test']
        assert fold_idx < 10
        
        self.dataset_root = path
        self.dataset_path = os.path.join(self.dataset_root, dataset)
        self.fold_idx = fold_idx        
        self.split = split
        self.x = None
        self.y = None
        self.cat = category 
        self.load()

    def load(self):
        file = np.load(self.dataset_path, allow_pickle=True)
        X = file['X']
        y = file['y']
        if self.split == 'train':
            fold_split =  file['folds'][self.fold_idx][0]
            
        else:
            fold_split =  file['folds'][self.fold_idx][1]
            
        self.x = X[fold_split]
        self.y = y[fold_split].argmax(1)
        lista_2=[]
        for i in range(len(self.x)):
            x_1 = self.x[i]
            x_1=np.concatenate(x_1, axis=0)
            list_1=[]
            for j in range(50):
                x_2=x_1[j]
                x_3 = x_2[:3] 
                list_1.append(x_3)
            lista_2.append(list_1)

        self.x = np.array(lista_2)
        #self.y = y

        restr= (self.y == self.cat)
        self.x = self.x[restr]
        self.y = self.y[restr]
        return self.x, self.y
    def __getitem__(self, idx):
        vec = self.x[idx]
        if self.category != -1:
            label = self.category
        else:
            label = self.y[idx].argmax()
        
        return vec[0].astype(np.float32), label

    

    def __len__(self):
        return len(self.x)



def correcting_size(path, cat, fold, path2save):
    for k in tqdm(range(1,31)):
        listing = [i for i in range(1,31) if i!= k]
        data = np.load(path+f'/cat{cat}-fold{fold}-{k}.npy',allow_pickle=True)
        print(data.shape, 'before')
        if data.shape[1]==3:
            if len(data)>50:
                new_data = data[:50,:]
                print(new_data.shape, 'after')
            elif 30<=len(data)<50:
                new_data = np.concatenate([data, data[-(50-len(data)):, :]])
                print(new_data.shape, 'after')
            elif len(data)< 30:
                for j in listing:
                    data2 = np.load(path+f'/cat{cat}-fold{fold}-{j}.npy',allow_pickle=True)
                    if len(data2)>50:
                        data2 = data2
                        break
                    else:
                        continue
                new_data = np.concatenate([data, data2[-(50-len(data)):, :]])
                print(new_data.shape, 'after')
            elif len(data)==50:
                new_data = data
            np.save(path2save+f'/cat{cat}-fold{fold}-{k}.npy', new_data)
            print('saved')
        else:
            continue
    print('FINISHED')
def zipping_all(path, cat, fold, path2save):
    all = []
    missing = 0
    for k in tqdm(range(1,31)):
        try:
            data = np.load(path+f'/cat{cat}-fold{fold}-{k}.npy',allow_pickle=True)
            print(data.shape)
            all.append(data)
        except:
            missing+=1
        
    comp = np.array(all)
    print('MISSING:',missing)
    add = random.sample(range(len(comp)+1), missing)
    for sample in add:
        chose = comp[sample]
        all.append(chose)
    np.save(path2save+f'/cat{cat}-fold{fold}.npy',np.array(all))
    print(np.array(all).shape)
    print('FINISHED')


def adding_synthetic(fold, dataset, path):
    all = []
    lab = []
    if dataset=='MHEALTH.npz':
        for cat  in range(12):
            data = np.load(path+f'/cat{cat}-fold{fold}.npy',allow_pickle=True)
            all.append(data)
            lab.append([cat]*len(data))
    else:
        for cat  in range(6):
            data = np.load(path+f'/cat{cat}-fold{fold}.npy',allow_pickle=True)
            all.append(data)
            lab.append([cat]*len(data))
    return np.concatenate(np.array(all)),  np.concatenate(np.array(lab))



def one_hot_encode(y):
    """
    Converte uma matriz de rótulos em uma matriz one-hot encoded.

    Parâmetros:
    y : array-like, shape (n_samples,)
        Vetor de rótulos a serem codificados.

    Retorna:
    y_one_hot : array, shape (n_samples, n_classes)
        Matriz one-hot encoded.
    """
    y = np.array(y)
    n_classes = np.max(y) + 1  # Assume que os rótulos são inteiros começando de 0
    y_one_hot = np.eye(n_classes)[y]

    return y_one_hot



def adding_cat_synthetic(fold,cat,amount,  path):
    all = []
    lab = []
    data = np.load(path+f'/cat{cat}-fold{fold}.npy',allow_pickle=True)
    #print(data.shape)
    add = random.sample(range(len(data)), amount)
    for sample in add:
        chose = data[sample]
        all.append(chose)
    lab.append([cat]*amount)
    return np.array(all),  np.concatenate(np.array(lab))




def TRTS(fold,dataset, path_real,pathfinal):
    all =[]
    lab = []
    if dataset=='MHEALTH.npz':
        for j in range(12):
            d = DataCat(split='test',fold_idx=fold,category= j,  dataset=dataset,path = path_real)
            X_real, y_real = d.load()
            amount = len(X_real)
            #print(amount, 'cat', j )
            x,y = adding_cat_synthetic(fold=fold,cat=j,amount=len(X_real),  path=pathfinal )
            all.append(x)
            lab.append(y)
    else:
        for j in range(6):
            d = DataCat(split='test',fold_idx=fold,category= j,  dataset=dataset,path = path_real)
            X_real, y_real = d.load()
            amount = len(X_real)
            #print(amount, 'cat', j )
            x,y = adding_cat_synthetic(fold=fold,cat=j,amount=len(X_real),  path=pathfinal )
            all.append(x)
            lab.append(y)
    return np.concatenate(all), np.concatenate(lab)



def augmenting(amount,fold,dataset, path_synthetic):
    data_all = []
    labels_all = []
    if dataset=='MHEALTH.npz':
        for cat in range(12):
            xx,yy = adding_cat_synthetic(fold=fold,cat=cat,amount=amount,  path=path_synthetic )
            data_all.append(xx)
            labels_all.append(yy)
    else:
        for cat in range(6):
            xx,yy = adding_cat_synthetic(fold=fold,cat=cat,amount=amount,  path=path_synthetic )
            data_all.append(xx)
            labels_all.append(yy)
    return np.concatenate(np.array(data_all)), np.concatenate(np.array(labels_all))