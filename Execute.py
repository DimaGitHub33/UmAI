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

## 1) Load Data -------------------------------------------------------------------------------
makeClassificationX,makeClassificationY = make_classification(n_samples=40000,n_features=10,class_sep=3)
XTrain, XTest, YTrain, YTest = train_test_split(makeClassificationX, makeClassificationY, test_size=0.3, random_state=0)

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
PC = PriditClassifier(pd.DataFrame(XTrain), conf={},logger  = logger)
priditScore,F,firstEigenVector  = PC.Pridit()

## 3) Model ----------------------------------------------------------------------------------
## Creating the Data 
Data = pd.DataFrame(XTrain)
Data['priditScore'] = priditScore
Data['priditScore'].describe()

Data['NewHistoricalYTarget'] = np.where(Data['priditScore'] >= Data['priditScore'].mean(),1,0)
Data['ActualY'] = pd.DataFrame(YTrain)

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

Output['Y'] = pd.DataFrame(YTrain)
Output.groupby('Target')['PredictGBM'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)
Output.groupby('Y')['PredictGBM'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)
Output.groupby('Rank')['Y'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)
Output.groupby('Rank')['PredictGBM'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)


## 4) Predict ----------------------------------------------------------------------------------
NewData = pd.concat([pd.DataFrame(XTest),pd.DataFrame({'Y':YTest})],axis=1)

conf={
    'Path':'/Users/dhhazanov/UmAI/Models/Model.pckl'##Where is the model saved
}
Predictions = Predict(NewData,conf,logger).Predict()
Predictions.describe()

Predictions['ActualY'] = NewData['Y'] 

Predictions.groupby('ActualY')['PredictGBM'].apply(np.mean).reset_index()
Predictions.groupby('Rank')['ActualY'].apply(np.mean).reset_index()
Predictions.groupby('Rank')['PredictGBM'].apply(np.mean).reset_index()
