

## ------------------------------------------------------------------------------------------------
## ---------------------------------------- Incurance Data ----------------------------------------
## ------------------------------------------------------------------------------------------------

## 0) Import Packages -------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from Pridit import PriditClassifier
from Functions import Ranks_Dictionary, RJitter, FunFactorYMC, FunNumericYMC
from Model import Model
from Predict import Predict
import logging as logger

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

from sklearn.ensemble import IsolationForest

## 1) Load Data -------------------------------------------------------------------------------------     
insuranceClaims = pd.read_csv("insurance_claims.csv")
insuranceClaims = pd.DataFrame(insuranceClaims)

XTrain, XTest = train_test_split(insuranceClaims, test_size=0.3, random_state=0)


## 2) Pridit Score ----------------------------------------------------------------------------------
## Creating the Data
Data = pd.DataFrame(XTrain)
Data.columns = Data.columns.astype(str)

## Creating the configuration
conf = {
    # 'UsingFacotr': 'OnlyVariables',  ##Both, OnlyVariables, None
    #'FactorVariables': FactorVariables,  ##List, None
    #'NumericVariables': NumericVariables,  ##list, None
    #'FactorVariables': [],  ##List, None
    #'NumericVariables': [],  ##list, None
    #'FactorsVariablesOrder': None,  ##List, None
    'NumericVariablesOrder': None  ##List, None
}

## Pridit
PC = PriditClassifier(Data = Data, conf = conf,logger = logger)
priditScore,F,firstEigenVector  = PC.Pridit()
pd.DataFrame(firstEigenVector).describe()

## 3) Model ----------------------------------------------------------------------------------
## Creating the Data 
Data = pd.DataFrame(XTrain)
Data['priditScore'] = priditScore
Data['priditScore'].describe()

## Creating the new unsupercised Y and adding the Actual Y
Data['NewHistoricalYTarget'] = np.where(Data['priditScore'] >= Data['priditScore'].mean(),0,1)


## Creating the configuration
conf={
    'Path':'/Users/dhhazanov/UmAI/Models/Model.pckl',
    'Target':'NewHistoricalYTarget',
    'ColumnSelectionType': 'Drop',#Drop,Keep
    'Keep': None,#['GENDER', 'FAMILY_STATUS','GIL'],
    'Drop': ['priditScore','policy_deductable2'],#['GIL','ISUK_MERAKEZ','FAMILY_STATUS','ISHUN','M_CHOD_TASHLOM_BR'],#None,
    'ModelType': None #GBM,Linear regression,...
}
RunModel = Model(Data,conf,logger)
Output = RunModel.fit()

## Checking the Model
Output.groupby('Target')['PredictGBM'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)
#    Target  PredictGBM
# 0       0        0.04
# 1       1        0.97

Output.groupby('Rank')['PredictGBM'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)
#    Rank  PredictGBM
# 0     2        0.00
# 1     3        0.00
# 2     4        0.00
# 3     5        0.29
# 4     6        0.93
# 5     7        1.00
# 6     8        1.00
# 7     9        1.00
# 8    10        1.00

## 4) Predict ----------------------------------------------------------------------------------
NewData = XTest

conf={
    'Path':'/Users/dhhazanov/UmAI/Models/Model.pckl'##Where is the model saved
}
PredictClass = Predict(NewData,conf,logger)
PredictClass.load_model(Path = '/Users/dhhazanov/UmAI/Models/conf3')##Path is where to write the configuration
Flag,Difference = PredictClass.pre_predict_validation()
Predictions = PredictClass.Predict()

Predictions.describe()


Predictions.groupby('Rank')['PredictGBM'].apply(np.mean).reset_index()
#    Rank  PredictGBM
# 0     6        0.00
# 1     7        0.10
# 2     8        0.92
# 3     9        1.00
# 4    10        1.00
