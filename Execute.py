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
makeClassificationX,makeClassificationY = make_classification(n_samples=40000,n_features = 15,class_sep = 4,weights = [0.8])
XTrain, XTest, YTrain, YTest = train_test_split(makeClassificationX, makeClassificationY, test_size=0.3, random_state=0)
                           
# makeClassificationX,makeClassificationY = load_breast_cancer(return_X_y=True)
# XTrain, XTest, YTrain, YTest = train_test_split(makeClassificationX, makeClassificationY, test_size=0.3, random_state=0)
## 2) Pridit Score ----------------------------------------------------------------------------------

## Creating the Data
Data = pd.DataFrame(XTrain)
Data.columns = Data.columns.astype(str)

## Creating the NumericVariablesOrder
NumericVariablesOrder = pd.DataFrame()
for col in Data.columns:
    if (np.abs(np.corrcoef(Data.loc[:,col],YTrain)[0,1])<=0.05):
        Data = Data.drop(col,axis=1)
        continue
    Row = pd.DataFrame(data={'Variable': col,
                             'Order': np.where(np.corrcoef(Data.loc[:,col],YTrain)[0,1]>=0,1,0)}, index=[0])
    print(np.corrcoef(Data.loc[:,col],YTrain)[0,1])
    NumericVariablesOrder = pd.concat([NumericVariablesOrder,Row])

## Creating the configuration
conf = {
    #'UsingFacotr': 'OnlyVariables',  ##Both, OnlyVariables, None
    #'FactorVariables': FactorVariables,  ##List, None
    #'NumericVariables': NumericVariables,  ##list, None
    #'FactorVariables': [],  ##List, None
    #'NumericVariables': [],  ##list, None
    #'FactorsVariablesOrder': None,  ##List, None
    'NumericVariablesOrder': NumericVariablesOrder  ##List, None
}

## Pridit
PC = PriditClassifier(Data = Data, conf = conf,logger = logger)
#PC = PriditClassifier(Data = Data, conf = {},logger = logger)
priditScore,F,firstEigenVector  = PC.Pridit()
pd.DataFrame(firstEigenVector).describe()

## 3) Model ----------------------------------------------------------------------------------
## Creating the Data 
Data = pd.DataFrame(XTrain)
Data['priditScore'] = priditScore
Data['priditScore'].describe()

## Creating the new unsupercised Y and adding the Actual Y
Data['NewHistoricalYTarget'] = np.where(Data['priditScore'] >= Data['priditScore'].quantile(0.80),0,1)
Data['NewHistoricalYTarget'] = np.where(Data['priditScore'] >= Data['priditScore'].mean(),0,1)

Data['ActualY'] = pd.DataFrame(YTrain)
pd.DataFrame(Data['ActualY'] == Data['NewHistoricalYTarget']).astype(float).describe()

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
#    Y  PredictGBM
# 0  0        0.38
# 1  1        0.98
Output.groupby('Rank')['Y'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)
#    Rank    Y
# 0     3 0.00
# 1     4 0.01
# 2     5 0.02
# 3     6 0.47
# 4     7 0.41
# 5     8 0.40
# 6    10 0.37
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
