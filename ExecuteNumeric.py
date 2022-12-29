## 0) Import Packages -------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from Pridit import PriditClassifier
from Functions import Ranks_Dictionary, RJitter, FunFactorYMC, FunNumericYMC
from NumericModel import NumericModel
from NumericPredict import NumericPredict
import logging as logger

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

## 1) Load Data -------------------------------------------------------------------------------------
makeRegressionX,makeRegressionY = make_regression(n_samples=40000,n_features = 10)
XTrain, XTest, YTrain, YTest = train_test_split(makeRegressionX, makeRegressionY, test_size=0.3, random_state=0)

#% --------------------------------------                 
#pd.DataFrame(YTrain).plot.hist(grid=True, bins=20, rwidth=0.9,color='#607c8e')

## 2) Model ----------------------------------------------------------------------------------
## Creating the Data 
Data = pd.DataFrame(XTrain)
Data['ActualY'] = pd.DataFrame(YTrain)

## Creating the configuration
conf={
    'Path':'/Users/dhhazanov/UmAI/Models/NumericModel.pckl',
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
# Check.describe()
#        Rank       Y  PredictGBM  PredictQuantile  PredictCalibrated
# count 10.00   10.00       10.00            10.00              10.00
# mean   5.50   -1.54       -1.54            -1.63              -1.54
# std    3.03  184.29      184.83           183.05             183.01
# min    1.00 -316.78     -317.46          -314.62            -314.85
# 25%    3.25 -108.86     -109.61          -108.23            -107.98
# 50%    5.50   -0.64       -0.43            -0.76              -0.49
# 75%    7.75  106.99      107.55           106.35             106.32
# max   10.00  309.70      310.40           307.31             307.49

## 3) Predict ----------------------------------------------------------------------------------
NewData = pd.concat([pd.DataFrame(XTest),pd.DataFrame({'Y':YTest})],axis=1)

conf={
    'Path':'/Users/dhhazanov/UmAI/Models/NumericModel.pckl'##Where is the model saved
}

PredictClass = NumericPredict(NewData,conf,logger)
PredictClass.get_conf_from_pkl(path = conf['Path'])
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
#        Rank  ActualY  PredictGBM  PredictQuantile  PredictCalibrated
# count 10.00    10.00       10.00            10.00              10.00
# mean   5.50    -0.16       -0.28            -0.21              -0.24
# std    3.03   144.37      145.15           143.28             142.94
# min    1.00  -246.54     -248.11          -244.61            -244.04
# 25%    3.25   -84.67      -85.68           -84.14             -83.90
# 50%    5.50    -0.22       -0.25            -0.09              -0.40
# 75%    7.75    85.82       86.74            85.01              84.65
# max   10.00   242.85      243.13           241.04             240.79


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
#        Rank        Y  PredictGBM  PredictQuantile  PredictCalibrated
# count 10.00    10.00       10.00            10.00              10.00
# mean   5.50  7445.03     7451.56          7501.95            7445.03
# std    3.03  4702.64     4480.54          4456.64            4454.74
# min    1.00   698.57      417.71           647.09             669.46
# 25%    3.25  4852.96     4896.28          5307.10            5087.09
# 50%    5.50  6977.07     7589.74          7308.61            7451.62
# 75%    7.75 10603.36    10198.42         10322.96           10045.70
# max   10.00 14628.00    14671.18         14383.73           14543.31

## 3) Predict ----------------------------------------------------------------------------------
NewData = pd.DataFrame(XTest).reset_index(drop=True)

conf={
    'Path':'/Users/dhhazanov/UmAI/Models/Model.pckl'##Where is the model saved
}

PredictClass = NumericPredict(NewData,conf,logger)
PredictClass.get_conf_from_pkl(path = conf['Path'])##Path is where to write the configuration
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
#        Rank  ActualY  PredictGBM  PredictQuantile  PredictCalibrated
# count 10.00    10.00       10.00            10.00              10.00
# mean   5.50  7561.53     7478.18          7450.40            7398.10
# std    3.03  4730.48     4485.66          4291.74            4234.51
# min    1.00   691.18      426.18           598.70             623.83
# 25%    3.25  5205.29     4936.12          5517.30            5452.77
# 50%    5.50  7197.02     7639.71          7522.93            7661.91
# 75%    7.75 10756.18    10279.07         10291.22           10127.69
# max   10.00 14921.43    14718.33         13986.97           13847.02