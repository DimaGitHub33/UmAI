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

RunModel.get_modelMetricsEstimation_from_pkl(conf['Path'])
#       Acuumalative Number Of People  # In Each Group  Percent  # Total
# <10                            1632          1632.00    58.29     2800
# <20                            2077           445.00    15.89     2800
# <30                            2295           218.00     7.79     2800
# <40                            2415           120.00     4.29     2800
# <50                            2499            84.00     3.00     2800
# <60                            2557            58.00     2.07     2800
# <70                            2595            38.00     1.36     2800
# <80                            2615            20.00     0.71     2800
# <90                            2642            27.00     0.96     2800
# <100                           2660            18.00     0.64     2800
# <200                           2736            76.00     2.71     2800
# <300                           2756            20.00     0.71     2800
# <400                           2770            14.00     0.50     2800
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
# mean   5.50     0.73        1.15             1.22               0.97
# std    3.03   181.67      181.75           180.27             179.86
# min    1.00  -306.37     -306.02          -302.54            -302.69
# 25%    3.25  -106.17     -105.58          -104.74            -104.28
# 50%    5.50     0.12        0.84             0.65               0.75
# 75%    7.75   107.75      107.57           107.58             106.77
# max   10.00   308.68      309.69           307.41             306.81


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

##Mappe
RunModel.get_modelMetricsEstimation_from_pkl(conf['Path'])
#       Acuumalative Number Of People  # In Each Group  Percent  # Total
# <10                              18            18.00    25.71       70
# <20                              32            14.00    20.00       70
# <30                              43            11.00    15.71       70
# <40                              53            10.00    14.29       70
# <50                              59             6.00     8.57       70
# <60                              61             2.00     2.86       70
# <70                              61             0.00     0.00       70
# <80                              62             1.00     1.43       70
# <90                              63             1.00     1.43       70
# <100                             65             2.00     2.86       70
# <200                             67             2.00     2.86       70
# <300                             69             2.00     2.86       70
# <400                             69             0.00     0.00       70
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
# mean   5.50  7588.19     7465.56          7476.66            7418.98
# std    3.03  4711.02     4534.80          4240.96            4250.63
# min    1.00   695.31      285.78           616.48             636.01
# 25%    3.25  5315.31     4834.65          5659.31            5456.47
# 50%    5.50  7104.52     7591.00          7566.60            7658.12
# 75%    7.75 10609.04    10403.08         10509.47           10426.51
# max   10.00 14770.45    14669.82         13863.47           13910.70