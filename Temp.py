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
breast_cancer_x,breast_cancer_y = load_breast_cancer(return_X_y=True)

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

## 1) Load Data -------------------------------------------------------------------------------------
dataHealth = pd.read_csv("/Users/dhhazanov/Downloads/Eli_data_health.csv")
dataHealth = pd.DataFrame(dataHealth)
dataHealth = dataHealth.drop(['Unnamed: 0','Column1'],axis=1)
## 2) Pridit Score ----------------------------------------------------------------------------------
# conf = {
#     'UsingFacotr': 'OnlyVariables',  ##Both, OnlyVariables, None
#     'FactorVariables': FactorVariables,  ##List, None
#     'NumericVariables': NumericVariables,  ##list, None
#     #'FactorVariables': [],  ##List, None
#     #'NumericVariables': [],  ##list, None
#     'FactorsVariablesOrder': None,  ##List, None
#     'NumericVariablesOrder': None  ##List, None
# }

PC = PriditClassifier(pd.DataFrame(dataHealth), conf={},logger=logger)
priditScore,F,firstEigenVector  = PC.Pridit()

## 3) Model ----------------------------------------------------------------------------------
## Creating the Data 
Data = pd.DataFrame(dataHealth)
Data['priditScore'] = priditScore
Data['priditScore'].describe()

## Creating the new unsupercised Y and adding the Actual Y
Data['NewHistoricalYTarget'] = np.where(Data['priditScore'] >= Data['priditScore'].mean(),0,1)
pd.DataFrame(Data['HAVE_MISHPAT_BR_2Y'].fillna(0) == Data['NewHistoricalYTarget']).astype(float).describe()

## Creating the configuration
conf={
    'Path':'/Users/dhhazanov/UmAI/Models/Model.pckl',
    'Target':'NewHistoricalYTarget',
    'ColumnSelectionType': 'Drop',#Drop,Keep
    'Keep': None,#['GENDER', 'FAMILY_STATUS','GIL'],
    'Drop': ['priditScore','ActualY'],#['GIL','ISUK_MERAKEZ','FAMILY_STATUS','ISHUN','M_CHOD_TASHLOM_BR'],#None,
    'ModelType': None #GBM,Linear regression,...
}
RunModel = Model(Data,conf,logger)
Output = RunModel.fit()

## Checking the Model
Output['Y'] = pd.DataFrame(YTrain)
Output.groupby('Target')['PredictGBM'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)
Output.groupby('Y')['PredictGBM'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)
Output.groupby('Rank')['Y'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)
Output.groupby('Rank')['PredictGBM'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)


## 3) Predict ----------------------------------------------------------------------------------
make_classification_x,make_classification_y = make_classification(n_samples=20000)
NewData = pd.concat([pd.DataFrame(make_classification_x),pd.DataFrame({'Y':make_classification_y})],axis=1)

conf={
    'Path':'/Users/dhhazanov/UmAI/Models/Model.pckl'##Where is the model saved
}
NewData.columns = NewData.columns.astype(str)
Predictions = Predict(NewData,conf).Predict()
Predictions.describe()

NewData['Predictions'] = Predictions['predictGBM']

AggregationTable = NewData.groupby('Y')['Predictions'].apply(np.mean).reset_index()
AggregationTable