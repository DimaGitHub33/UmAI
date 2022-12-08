
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

from sklearn.ensemble import IsolationForest

## 1) Load Data -------------------------------------------------------------------------------------
makeClassificationX,makeClassificationY = make_classification(n_samples=40000,n_features = 15,class_sep = 4,weights = [0.99])
XTrain, XTest, YTrain, YTest = train_test_split(makeClassificationX, makeClassificationY, test_size=0.3, random_state=0)
                           
# makeClassificationX,makeClassificationY = load_breast_cancer(return_X_y=True)
# XTrain, XTest, YTrain, YTest = train_test_split(makeClassificationX, makeClassificationY, test_size=0.3, random_state=0)
## 2) Pridit Score ----------------------------------------------------------------------------------

## Creating the Data
Data = pd.DataFrame(XTrain)
Data.columns = Data.columns.astype(str)

clf = IsolationForest(random_state = 0, n_jobs = 4,).fit(Data)
Scores = np.where(clf.predict(Data)<=0,1,0)

## 3) Model ----------------------------------------------------------------------------------
## Creating the Data 
Data = pd.DataFrame(XTrain)
Data['Scores'] = Scores
Data['Scores'].describe()

## Creating the new unsupercised Y and adding the Actual Y
Data['NewHistoricalYTarget'] = Data['Scores']

Data['ActualY'] = pd.DataFrame(YTrain)
pd.DataFrame(Data['ActualY'] == Data['NewHistoricalYTarget']).astype(float).describe()

## Creating the configuration
conf={
    'Path':'/Users/dhhazanov/UmAI/Models/Model.pckl',
    'Target':'NewHistoricalYTarget',
    'ColumnSelectionType': 'Drop',#Drop,Keep
    'Keep': None,#['GENDER', 'FAMILY_STATUS','GIL'],
    'Drop': ['Scores','ActualY'],#['GIL','ISUK_MERAKEZ','FAMILY_STATUS','ISHUN','M_CHOD_TASHLOM_BR'],#None,
    'ModelType': None #GBM,Linear regression,...
}
RunModel = Model(Data,conf,logger)
Output = RunModel.fit()

## Checking the Model
Output['Y'] = pd.DataFrame(YTrain)
Output.groupby('Target')['PredictGBM'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)
#    Target  PredictGBM
# 0       0        0.01
# 1       1        0.83
Output.groupby('Y')['PredictGBM'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)
#    Y  PredictGBM
# 0  0        0.04
# 1  1        0.57
Output.groupby('Rank')['Y'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)
#    Rank    Y
# 0     6 0.00
# 1     7 0.01
# 2     8 0.01
# 3     9 0.01
# 4    10 0.10
Output.groupby('Rank')['PredictGBM'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)
#    Rank  PredictGBM
# 0     6        0.00
# 1     7        0.00
# 2     8        0.01
# 3     9        0.04
# 4    10        0.44

## 4) Predict ----------------------------------------------------------------------------------
NewData = pd.concat([pd.DataFrame(XTest),pd.DataFrame({'Y':YTest})],axis=1)

conf={
    'Path':'/Users/dhhazanov/UmAI/Models/Model.pckl'##Where is the model saved
}
Predictions = Predict(NewData,conf,logger).Predict()
Predictions.describe()

Predictions['ActualY'] = NewData['Y'] 

Predictions.groupby('ActualY')['PredictGBM'].apply(np.mean).reset_index()
#    ActualY  PredictGBM
# 0        0        0.04
# 1        1        0.43
Predictions.groupby('Rank')['ActualY'].apply(np.mean).reset_index()
#    Rank  ActualY
# 0     6     0.01
# 1     7     0.01
# 2     8     0.00
# 3     9     0.01
# 4    10     0.09
Predictions.groupby('Rank')['PredictGBM'].apply(np.mean).reset_index()
#    Rank  PredictGBM
# 0     6        0.00
# 1     7        0.00
# 2     8        0.02
# 3     9        0.07
# 4    10        0.34
Predictions.groupby('Rank')['ActualY'].apply(np.mean).reset_index()
#    Rank  ActualY
# 0     6     0.01
# 1     7     0.01
# 2     8     0.00
# 3     9     0.01
# 4    10     0.09
