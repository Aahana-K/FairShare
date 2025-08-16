
from flask import Flask, request,jsonify

from flask_cors import CORS

import torch 
import torch.nn as nn

import torch.optim as optim

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from math import sqrt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

import json
from torch.autograd import Variable
import joblib
import os

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
    
pAth = os.path.dirname(os.path.abspath(__file__))
cDegree = joblib.load(os.path.join(pAth,'codedDegree.pkl'))
cField = joblib.load(os.path.join(pAth,'codedField.pkl'))
cLevel = joblib.load(os.path.join(pAth,'codedLevel.pkl'))
cSize = joblib.load(os.path.join(pAth,'codedSize.pkl'))
cCountry = joblib.load(os.path.join(pAth,'codedCountry.pkl'))


fairPredAI = ANN(inputDim = 38,outputDim=1)
fairPredAI.load_state_dict(torch.load(os.path.join(pAth,'fairPredAI.pth'),map_location=torch.device('cpu')))
fairPredAI.eval()

appp = Flask(__name__)

CORS(appp)
@appp.route("/predict",methods=["POST"])

def predict():
    try:
        print("okayy")
        pulledData = request.get_json()
        print('datapullef')
        data = pd.DataFrame(pulledData['userInputs'])

        trDegree = cDegree.transform(data[['degree']])
        trField = cField.transform(data[['field']])
        trLevel = cLevel.transform(data[['job_level']])
        trSize = cSize.transform(data[['company_size']])
        trCountry = cCountry.transform(data[['country']])
        degreeDF= pd.DataFrame(trDegree,columns = cDegree.get_feature_names_out(['degree']))
        fieldDF= pd.DataFrame(trField,columns = cField.get_feature_names_out(['field']))
        countryDF= pd.DataFrame(trCountry,columns = cCountry.get_feature_names_out(['country']))
        levelDF= pd.DataFrame(trLevel,columns = ['job_level'])
        sizeDF= pd.DataFrame(trSize,columns =['company_size'])

        fds = pd.concat([data.drop(columns = ['degree','field','country','job_level','company_size']),degreeDF,fieldDF,countryDF,levelDF,sizeDF],axis=1)

        numCols = ['hours_daily','yrs_experience']
        for col in numCols:
            fds[col] = fds[col].astype(float)
        x= torch.tensor(fds.values,dtype=torch.float32)
        print("hmm")
        with torch.no_grad():
            prediction = fairPredAI(x)
        pprediction = prediction.tolist()
        print(pprediction)
        '''
        if isinstance(pprediction,float):
            pprediction = [prediction]
        pprediction = [round(p) for p in pprediction]
        print("HII")
        print(prediction)
        '''
        #pprediction = [round(p) for p in pprediction]
        return jsonify({"prediction":pprediction})
    except Exception as e:
        return jsonify({"error":str(e)})
    
if __name__ == "__main__":
    appp.run(debug=True, use_reloader = False)
