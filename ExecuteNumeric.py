## 0) Import Packages -------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from Pridit import PriditClassifier
from Functions import Ranks_Dictionary, RJitter, FunFactorYMC, FunNumericYMC
from NumericModel import NumericModel
from NumericPredict import NumericPredict
import logging as logger

from sklearn.datasets import load_breast_cancer
breast_cancer_x,breast_cancer_y = load_breast_cancer(return_X_y=True)

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

## 1) Load Data -------------------------------------------------------------------------------------
makeRegressionX,makeRegressionY = make_regression(n_samples=40000,n_features = 10)
XTrain, XTest, YTrain, YTest = train_test_split(makeRegressionX, makeRegressionY, test_size=0.3, random_state=0)
                        

## 2) Model ----------------------------------------------------------------------------------
## Creating the Data 
Data = pd.DataFrame(XTrain)
Data['ActualY'] = pd.DataFrame(YTrain)

## Creating the configuration
conf={
    'Path':'/Users/dhhazanov/UmAI/Models/Model.pckl',
    'Target':'ActualY',
    'ColumnSelectionType': 'Drop',#Drop,Keep
    'Keep': None,#['GENDER', 'FAMILY_STATUS','GIL'],
    'Drop': ['priditScore'],#['GIL','ISUK_MERAKEZ','FAMILY_STATUS','ISHUN','M_CHOD_TASHLOM_BR'],#None,
    'ModelType': None #GBM,Linear regression,...
}
RunModel = NumericModel(Data,conf,logger)
Output = RunModel.fit()
Output.describe()

## Checking the Model
Output['Y'] = pd.DataFrame(YTrain)
a = Output.groupby('Rank')['Y'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)
b = Output.groupby('Rank')['PredictGBM'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)
c = Output.groupby('Rank')['PredictQuantile'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)
d = Output.groupby('Rank')['PredictCalibrated'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)

a = pd.concat([a.reset_index(drop=True), b.drop('Rank',axis=1)], axis=1)
a = pd.concat([a.reset_index(drop=True), c.drop('Rank',axis=1)], axis=1)
Check = pd.concat([a.reset_index(drop=True), d.drop('Rank',axis=1)], axis=1)
del a,b,c,d
Check.describe()

## 3) Predict ----------------------------------------------------------------------------------
NewData = pd.concat([pd.DataFrame(XTest),pd.DataFrame({'Y':YTest})],axis=1)

conf={
    'Path':'/Users/dhhazanov/UmAI/Models/Model.pckl'##Where is the model saved
}

PredictClass = NumericPredict(NewData,conf,logger)
PredictClass.load_model(Path = '/Users/dhhazanov/UmAI/Models/conf2')##Path is where to write the configuration
Flag,Difference = PredictClass.pre_predict_validation()
Predictions = PredictClass.Predict()
Predictions.describe()

Predictions['ActualY'] = NewData['Y'] 

a = Predictions.groupby('Rank')['ActualY'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)
b = Predictions.groupby('Rank')['PredictGBM'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)
c = Predictions.groupby('Rank')['PredictQuantile'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)
d = Predictions.groupby('Rank')['PredictCalibrated'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)

a = pd.concat([a.reset_index(drop=True), b.drop('Rank',axis=1)], axis=1)
a = pd.concat([a.reset_index(drop=True), c.drop('Rank',axis=1)], axis=1)
Check = pd.concat([a.reset_index(drop=True), d.drop('Rank',axis=1)], axis=1)
del a,b,c,d
Check.describe()



## --------------------------------------------------------------------------------------------------
## -- Insurance data -------------------------------------------------------------------------------------
## --------------------------------------------------------------------------------------------------

## 1) Load Data -------------------------------------------------------------------------------------
insuranceClaims = pd.read_csv('/Users/dhhazanov/UmAI/Data/insurance_claims.csv')
XTrain, XTest = train_test_split(insuranceClaims,test_size=0.3, random_state=0)
                        

## 2) Model ----------------------------------------------------------------------------------
## Creating the Data 
Data = pd.DataFrame(XTrain).reset_index(drop=True)

## Creating the configuration
conf={
    'Path':'/Users/dhhazanov/UmAI/Models/Model.pckl',
    'Target':'injury_claim',
    'ColumnSelectionType': 'Drop',#Drop,Keep
    'Keep': None,#['GENDER', 'FAMILY_STATUS','GIL'],
    'Drop': ['priditScore'],#['GIL','ISUK_MERAKEZ','FAMILY_STATUS','ISHUN','M_CHOD_TASHLOM_BR'],#None,
    'ModelType': None #GBM,Linear regression,...
}
RunModel = NumericModel(Data,conf,logger)
Output = RunModel.fit()
Output.describe()

## Checking the Model
Output['Y'] = Data['injury_claim']
a = Output.groupby('Rank')['Y'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)
b = Output.groupby('Rank')['PredictGBM'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)
c = Output.groupby('Rank')['PredictQuantile'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)
d = Output.groupby('Rank')['PredictCalibrated'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)

a = pd.concat([a.reset_index(drop=True), b.drop('Rank',axis=1)], axis=1)
a = pd.concat([a.reset_index(drop=True), c.drop('Rank',axis=1)], axis=1)
Check = pd.concat([a.reset_index(drop=True), d.drop('Rank',axis=1)], axis=1)
del a,b,c,d
Check.describe()

## 3) Predict ----------------------------------------------------------------------------------
NewData = pd.DataFrame(XTest).reset_index(drop=True)

conf={
    'Path':'/Users/dhhazanov/UmAI/Models/Model.pckl'##Where is the model saved
}

PredictClass = NumericPredict(NewData,conf,logger)
PredictClass.load_model(Path = '/Users/dhhazanov/UmAI/Models/conf2')##Path is where to write the configuration
Flag,Difference = PredictClass.pre_predict_validation()
Predictions = PredictClass.Predict()
Predictions.describe()

Predictions['ActualY'] = NewData['injury_claim'] 

a = Predictions.groupby('Rank')['ActualY'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)
b = Predictions.groupby('Rank')['PredictGBM'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)
c = Predictions.groupby('Rank')['PredictQuantile'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)
d = Predictions.groupby('Rank')['PredictCalibrated'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)

a = pd.concat([a.reset_index(drop=True), b.drop('Rank',axis=1)], axis=1)
a = pd.concat([a.reset_index(drop=True), c.drop('Rank',axis=1)], axis=1)
Check = pd.concat([a.reset_index(drop=True), d.drop('Rank',axis=1)], axis=1)
del a,b,c,d
Check.describe()