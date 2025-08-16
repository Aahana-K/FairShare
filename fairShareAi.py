print("running startd")
import torch
import torch.nn as nn

import torch.optim as optim

import pandas as pd

import numpy as np

import sklearn
from sklearn.model_selection import train_test_split
from math import sqrt

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error



from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

#from torch.autograd import Varaible
#DBLE CHECK NOT MISSING ANY IMPORT STUFFF
import joblib
import os
import json

print("stufff importd")

pAth = os.path.dirname(os.path.abspath(__file__))

data=pd.read_csv(os.path.join(pAth,'mens_salaries_updated.csv'))

degreeE = OneHotEncoder(sparse_output=False)
fieldE = OneHotEncoder(sparse_output=False)
levelE = OrdinalEncoder(categories=[['entry','intermediate','senior']])
sizeE = OrdinalEncoder(categories=[['micro','small','medium','large']])
countryE = OneHotEncoder(sparse_output=False)

tDegree = degreeE.fit_transform(data[['degree']])
tField = fieldE.fit_transform(data[['field']])
tCountry = countryE.fit_transform(data[['country']])
tLevel = levelE.fit_transform(data[['job_level']])
tSize = sizeE.fit_transform(data[['company_size']])

degreeDF= pd.DataFrame(tDegree,columns = degreeE.get_feature_names_out(['degree']))
fieldDF= pd.DataFrame(tField,columns = fieldE.get_feature_names_out(['field']))
countryDF= pd.DataFrame(tCountry,columns = countryE.get_feature_names_out(['country']))
levelDF= pd.DataFrame(tLevel,columns = ['job_level'])
sizeDF= pd.DataFrame(tSize,columns =['company_size'])

fds = pd.concat([data.drop(columns = ['degree','field','country','job_level','company_size','salary_usd']),degreeDF,fieldDF,countryDF,levelDF,sizeDF],axis=1)

x= torch.tensor(fds.values,dtype=torch.float32)
y = torch.tensor(data['salary_usd'].values,dtype=torch.float32)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

trLoop = DataLoader(TensorDataset(x_train,y_train),batch_size=64,shuffle=True)
teLoop = DataLoader(TensorDataset(x_test,y_test),batch_size=64,shuffle=False)


print(x.shape)

class ANN(nn.Module):
    def __init__(self,inputDim=38,outputDim=1):
        super(ANN,self).__init__()
        self.layr1 = nn.Linear(inputDim,64)
        self.layr2 = nn.Linear(64,64)
        self.layr3 = nn.Linear(64,32)
        self.layr4 = nn.Linear(32,32)
        self.layr5 = nn.Linear(32,1)
        self.dropout = nn.Dropout(0.30)
    def forward(self,x):
        x = F.relu(self.layr1(x))
        x = F.relu(self.layr2(x))
        x = self.dropout(x)
        x = F.relu(self.layr3(x))
        x = F.relu(self.layr4(x))
        x = F.relu(self.layr5(x))
        return(x)
    
fairPredAI = ANN(inputDim = 38,outputDim=1)
optimizer= torch.optim.Adam(fairPredAI.parameters(),lr=0.005,weight_decay=1e-6)

print('MODEL CREATEDDDD')

epochs = 150
fairPredAI.train()

lossFunc = torch.nn.MSELoss()
for i in range(epochs):
    trainloss = 0.0

    for x,y in trLoop:
        optimizer.zero_grad()
        output=fairPredAI(x)
        loss = lossFunc(output,y.view(-1,1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(fairPredAI.parameters(),max_norm=1.0)
        optimizer.step()
        trainloss+= loss.item()*x.size(0)

    trainloss = trainloss/len(trLoop.dataset)
    print(f"Epoch: {i+1} \tTraining Loss: {trainloss:.4f}")


degreeCategories = degreeE.categories_[0].tolist()
fieldCategories = fieldE.categories_[0].tolist()
levelCategories = levelE.categories_[0].tolist()
sizeCategories = sizeE.categories_[0].tolist()
countryCategories = countryE.categories_[0].tolist()
'''
joblib.dump(degreeE,os.path.join(pAth,'codedDegree.pkl'))
joblib.dump(fieldE,os.path.join(pAth,'codedField.pkl'))
joblib.dump(levelE,os.path.join(pAth,'codedLevel.pkl'))
joblib.dump(sizeE,os.path.join(pAth,'codedSize.pkl'))
joblib.dump(countryE,os.path.join(pAth,'codedCountry.pkl'))
'''
torch.save(fairPredAI.state_dict(),os.path.join(pAth,'fairPredAI.pth'))

'''
torch.save(fairPredAI.state_dict(),os.path.join(pAth,'fairPredAI'))

with open('degreeCategories.json','w')as f:
    json.dump(degreeCategories,f)
with open('fieldCategories.json','w')as f:
    json.dump(fieldCategories,f)
with open('levelCategories.json','w')as f:
    json.dump(levelCategories,f)
with open('sizeCategories.json','w')as f:
    json.dump(sizeCategories,f)
with open('countryCategories.json','w')as f:
    json.dump(countryCategories,f)





'''
