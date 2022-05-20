import os
import math
import time
import random
import shutil
import albumentations
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter
import scipy as sp
from scipy.special import softmax
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm
from functools import partial
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import torchvision.models as models
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
import albumentations
from albumentations.pytorch import ToTensorV2
import timm
import streamlit as st
import warnings 
warnings.filterwarnings('ignore')
cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def random_seed(rseed=42):
  random.seed(rseed)
  os.environ['PYTHONHASHSEED'] = str(rseed)
  np.random.seed(rseed)
  torch.manual_seed(rseed)
  torch.cuda.manual_seed(rseed)
  torch.backends.cudnn.deterministic = True

random_seed(rseed=2020)
 
class CASSAVA:
  Workers = 8
  Model = 'resnext50_32x4d'
  Img_Size = 512
  BS = 32
  number_of_labels = 5
  KFolds = [0, 1, 2, 3, 4]


def resnext_aug(*, df):
  if df == 'valid':
    return albumentations.Compose([
      albumentations.Resize(CASSAVA.Img_Size, CASSAVA.Img_Size),
      albumentations.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
      ),
      ToTensorV2(),
    ])

DIR_PATH = './'
ResNext_Models_Path = 'models/'
if not os.path.exists(DIR_PATH):
  os.makedirs(DIR_PATH)
    
Cassava_Test_Img = 'input/buffer'

Img_Size = 512
efficient_aug = albumentations.Compose([
  albumentations.CenterCrop(Img_Size, Img_Size, p=1),
  albumentations.Resize(Img_Size, Img_Size),
  albumentations.Normalize()
])

EfficientNet_BS = 1
Efficient= ['tf_efficientnet_b5_ns'] * 5
EfficientNet_Model_Path = ['models/tf_efficientnet_b5_ns_kfold0_best.pth', 
              'models/tf_efficientnet_b5_ns_kfold1_best.pth', 
              'models/tf_efficientnet_b5_ns_kfold2_best.pth',
              'models/tf_efficientnet_b5_ns_kfold3_best.pth',
              'models/tf_efficientnet_b5_ns_kfold4_best.pth']

class ResNext_DataClass(Dataset):
  def __init__(self, data, transform=None):
    self.data = data
    self.image_ids = data['image_id'].values
    self.transform = transform
        
  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    image_ids_files = self.image_ids[idx]
    image_ids_path = f'{Cassava_Test_Img}/{image_ids_files}'
    cassava_img = cv2.imread(image_ids_path)
    cassava_img = cv2.cvtColor(cassava_img, cv2.COLOR_BGR2RGB)

    if self.transform:
      aug_img = self.transform(image=cassava_img)
      cassava_img = aug_img['image']

    return cassava_img

class EfficientNet_DataClass(Dataset):
  def __init__(self, data, method, transform=None):
    self.data = data.reset_index(drop=True)
    self.method = method
    self.transform = transform
        
  def __len__(self):
    return len(self.data)
    
  def __getitem__(self, index):
    cassava_item = self.data.loc[index]
    cassava_img = cv2.imread(cassava_item.filepath)
    cassava_img = cv2.cvtColor(cassava_img, cv2.COLOR_BGR2RGB)
        
    if self.transform is not None:
      aug_img = self.transform(image=cassava_img)
      cassava_img = aug_img['image']

    cassava_img = cassava_img.astype(np.float32)
    cassava_img = cassava_img.transpose(2,0,1)

    if self.method == 'test':
      cassava_img = torch.tensor(cassava_img).float()
      return cassava_img
    else:
      return torch.tensor(cassava_img).float(), torch.tensor(item.label).float()

class Cassava_ResNext50(nn.Module):
  def __init__(self, resnext50='resnext50_32x4d', pretrained=False):
    super().__init__()
    self.resnext50 = timm.create_model(resnext50, pretrained=pretrained)
    resnext50_in_features = self.resnext50.fc.in_features
    self.resnext50.fc = nn.Linear(resnext50_in_features, CASSAVA.number_of_labels)

  def forward(self, input):
    input = self.resnext50(input)
    return input

class Cassava_EfficientNet(nn.Module):
  def __init__(self, efficientnet='tf_efficientnet_b5_ns', pretrained=False):
    super().__init__()
    self.efficientnet = timm.create_model(efficientnet, pretrained=pretrained)
    efficientnet_in_features = self.efficientnet.classifier.in_features
    self.efficientnet.classifier = nn.Linear(efficientnet_in_features, CASSAVA.number_of_labels)

  def forward(self, input):
    input = self.efficientnet(input)
    return input

def model_dict_resnext(resnext_path):
  try:  
    model_dict = torch.load(resnext_path)['resnext50']
  except:  
    model_dict = torch.load(resnext_path)['resnext50']
    model_dict = {k[7:] if k.startswith('module.') else k: model_dict[k] for k in model_dict.keys()}

  return model_dict

def model_dict_efficient(model_path):
  try:  
    model_dict = torch.load(model_path)['efficientnet']     
  except:  
    model_dict = torch.load(model_path)['efficientnet']
    model_dict = {k[7:] if k.startswith('module.') else k: model_dict[k] for k in model_dict.keys()}

  return model_dict

def cassava_inf(cassava_model, cassava_model_load, cassava_test_iterable, cuda):
  cassava_model.to(cuda)
  tqdm_test = tqdm(enumerate(cassava_test_iterable), total=len(cassava_test_iterable))
  cassava_probability = []

  for index, (cassava_img) in tqdm_test:
    cassava_img = cassava_img.to(cuda)
    average_forecast = []

    for model_state in cassava_model_load:
      cassava_model.load_state_dict(model_state)
      cassava_model.eval()
      with torch.no_grad():
        disease_forecast = cassava_model(cassava_img)
      average_forecast.append(disease_forecast.softmax(1).to('cpu').numpy())
    average_forecast = np.mean(average_forecast, axis=0)
    cassava_probability.append(average_forecast)
  cassava_probability = np.concatenate(cassava_probability)
  return cassava_probability

def tta_cassava_inf(model,cassava_test_iterable):
  model.eval()
  cassava_tqdm = tqdm(cassava_test_iterable)
  cassava_probability = []

  with torch.no_grad():
    for index, cassava_img in enumerate(cassava_tqdm):
      img = cassava_img.to(cuda)
      img = torch.stack([img,img.flip(-1),img.flip(-2),img.flip(-1,-2),
      img.transpose(-1,-2),img.transpose(-1,-2).flip(-1),
      img.transpose(-1,-2).flip(-2),img.transpose(-1,-2).flip(-1,-2)],0)
      img = img.view(-1, 3, Img_Size, Img_Size)
      output = model(img)
      output = output.view(EfficientNet_BS, 8, -1).mean(1)
      cassava_probability += [torch.softmax(output, 1).detach().cpu()]            
    cassava_probability = torch.cat(cassava_probability).cpu().numpy()

  return cassava_probability

def final_fun(file_name):
  test = pd.read_csv('sample_submission.csv')
  test['image_id'] = file_name
  test['filepath'] = test.image_id.apply(lambda x: os.path.join('input/buffer', f'{x}'))

  data_class_test = EfficientNet_DataClass(test, 'test', transform=efficient_aug)
  iterable_test = torch.utils.data.DataLoader(data_class_test, batch_size=EfficientNet_BS, shuffle=False,  num_workers=4)

  model = Cassava_ResNext50(CASSAVA.Model, pretrained=False)
  cassava_model_load = [model_dict_resnext(ResNext_Models_Path+f'{CASSAVA.Model}_kfold{kfold}_best.pth') for kfold in CASSAVA.KFolds]
  cassava_test_data_class = ResNext_DataClass(test, transform=resnext_aug(df='valid'))
  cassava_test_iterable = DataLoader(cassava_test_data_class, batch_size=CASSAVA.BS, shuffle=False, 
                         num_workers=CASSAVA.Workers, pin_memory=True)
  forecast = cassava_inf(model, cassava_model_load, cassava_test_iterable, cuda)

  Efficient_Models_Path = 'models/'
  efficient_forecast = []
  cassava_model_load = [model_dict_efficient(Efficient_Models_Path+f'tf_efficientnet_b5_ns_kfold{kfold}_best.pth') for kfold in CASSAVA.KFolds]
  for index in range(len(Efficient)):
    model = Cassava_EfficientNet(efficientnet=Efficient[index], pretrained=False)
    model = model.to(cuda)
    model.load_state_dict(cassava_model_load[index])
    efficient_forecast += [tta_cassava_inf(model,iterable_test)]

  final_forecast = 0.5*forecast + 0.5*np.mean(efficient_forecast, axis=0)
  test['label'] = softmax(final_forecast).argmax(1)
  test = test[['image_id', 'label']]
  cassava_to_disease_id = pd.read_json('label_num_to_disease_map.json',orient='index')
  output_label = str(test['label'][0]) +" / "+ cassava_to_disease_id[0][test['label'][0]]
  return output_label

def save_uploaded_file(uploaded_file):
  try:
    with open(os.path.join('input/buffer',uploaded_file.name),'wb') as f:
        f.write(uploaded_file.getbuffer())
    return 1    
  except:
    return 0

def main():       

    cassava_box = """ 
    <div style ="background-color:blue;padding:5px"> 
    <h1 style ="color:black;text-align:center;">Project 3</h1>
    <h1 style ="color:black;text-align:center;">Cassava Leaf Disease Classification</h1>
    </div> 
    """
    st.markdown(cassava_box, unsafe_allow_html = True) 
    
    uploaded_file = st.file_uploader("Upload Image")
    if uploaded_file is not None:
        if save_uploaded_file(uploaded_file): 
            display_image = Image.open(uploaded_file)
            st.image(display_image)
            

    if st.button("Disease Classification"): 
        prediction = final_fun(uploaded_file.name)
        os.remove('input/buffer/'+uploaded_file.name)
        st.success(prediction)
     
if __name__=='__main__': 
    main()
