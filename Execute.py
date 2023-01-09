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
    if (np.abs(np.corrcoef(Data.loc[:,col],YTrain)[0,1])<=0.0005):
        Data = Data.drop(col,axis=1)
        continue
    Row = pd.DataFrame(data={'Variable': col,
                             'Order': np.where(np.corrcoef(Data.loc[:,col],YTrain)[0,1]>=0,1,0)}, index=[0])
    print(np.corrcoef(Data.loc[:,col],YTrain)[0,1])
    NumericVariablesOrder = pd.concat([NumericVariablesOrder,Row])
#NumericVariablesOrder = None
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
#           0
# count 13.00
# mean  -0.07
# std    0.28
# min   -0.63
# 25%   -0.01
# 50%   -0.00
# 75%   -0.00
# max    0.48
## 3) Model ----------------------------------------------------------------------------------
## Creating the Data 
Data = pd.DataFrame(XTrain)
Data['priditScore'] = priditScore
Data['priditScore'].describe()

## Creating the new unsupercised Y and adding the Actual Y
Data['NewHistoricalYTarget'] = np.where(Data['priditScore'] >= Data['priditScore'].quantile(0.10),0,1)
Data['NewHistoricalYTarget'] = np.where(Data['priditScore'] >= Data['priditScore'].mean(),0,1)

Data['ActualY'] = pd.DataFrame(YTrain)
pd.DataFrame(Data['ActualY'] == Data['NewHistoricalYTarget']).astype(float).describe()

## Creating the configuration
conf={
    'Path':'/Users/dhhazanov/UmAI/Models/Model.pckl',
    'ValidationDataPercent': 0.3,
    'Target':'NewHistoricalYTarget',
    'ColumnSelectionType': 'Drop',#Drop,Keep
    'Keep': None,#['GENDER', 'FAMILY_STATUS','GIL'],
    'Drop': ['priditScore','ActualY'],#['GIL','ISUK_MERAKEZ','FAMILY_STATUS','ISHUN','M_CHOD_TASHLOM_BR'],#None,
    'ModelType': None #GBM,Linear regression,...
}
RunModel = Model(Data,conf,logger)
Output = RunModel.fit()

RunModel.get_modelMetricsEstimation_from_pkl(conf['Path'])
#{'Accuracy': 0.9963095238095238, 'F1': 0.9963404556722937, 'Precision': 0.9957527135441245, 'Recall': 0.9969288920387432, 'RocCurve': (array([0.        , 0.00431965, 1.        ]), array([0.        , 0.99692889, 1.        ]), array([2, 1, 0]))}
## Checking the Model
Output['Y'] = pd.DataFrame(YTrain)
Output.groupby('Target')['PredictGBM'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)
Output.groupby('Y')['PredictGBM'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)
#    Y  PredictGBM
# 0  0        0.50
# 1  1        0.54
Output.groupby('Rank')['Y'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)
#    Rank    Y
# 0     1 0.01
# 1     2 0.01
# 2     3 0.00
# 3     4 0.01
# 4     5 0.04
# 5     6 0.02
# 6     7 0.01
# 7     8 0.01
# 8     9 0.02
# 9    10 0.02
Output.groupby('Rank')['PredictGBM'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)
#    Rank  PredictGBM
# 0     1        0.00
# 1     2        0.00
# 2     3        0.00
# 3     4        0.00
# 4     5        0.00
# 5     6        0.99
# 6     7        1.00
# 7     8        1.00
# 8     9        1.00
# 9    10        1.00

## 4) Predict ----------------------------------------------------------------------------------
NewData = pd.concat([pd.DataFrame(XTest),pd.DataFrame({'Y':YTest})],axis=1)

conf={
    'Path':'/Users/dhhazanov/UmAI/Models/Model.pckl'##Where is the model saved
}

PredictClass = Predict(NewData,conf,logger)
PredictClass.get_conf_from_pkl(path = conf['Path'])##Return the conf of the model
Flag,Difference = PredictClass.pre_predict_validation()
Predictions = PredictClass.Predict()
Predictions.describe()

Predictions['ActualY'] = NewData['Y'] 

Predictions.groupby('ActualY')['PredictGBM'].apply(np.mean).reset_index()
#    ActualY  PredictGBM
# 0        0        0.50
# 1        1        0.57
Predictions.groupby('Rank')['ActualY'].apply(np.mean).reset_index()
#    Rank  ActualY
# 0     1     0.00
# 1     2     0.01
# 2     3     0.01
# 3     4     0.01
# 4     5     0.03
# 5     6     0.02
# 6     7     0.01
# 7     8     0.02
# 8     9     0.02
# 9    10     0.01
Predictions.groupby('Rank')['PredictGBM'].apply(np.mean).reset_index()
#    Rank  PredictGBM
# 0     1        0.00
# 1     2        0.00
# 2     3        0.00
# 3     4        0.00
# 4     5        0.00
# 5     6        0.98
# 6     7        1.00
# 7     8        1.00
# 8     9        1.00
# 9    10        1.00







## ------------------------------------------------------------------------------------------------
## ----------------------------------------- breast cancer ----------------------------------------
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
makeClassificationX,makeClassificationY = load_breast_cancer(return_X_y=True)
XTrain, XTest, YTrain, YTest = train_test_split(makeClassificationX, makeClassificationY, test_size=0.3, random_state=0)

## 2) Pridit Score ----------------------------------------------------------------------------------
## Creating the Data
Data = pd.DataFrame(XTrain)
Data.columns = Data.columns.astype(str)

## Creating the NumericVariablesOrder
# NumericVariablesOrder = pd.DataFrame()
# for col in Data.columns:
#     if (np.abs(np.corrcoef(Data.loc[:,col],YTrain)[0,1])<=0.05):
#         Data = Data.drop(col,axis=1)
#         continue
#     Row = pd.DataFrame(data={'Variable': col,
#                              'Order': np.where(np.corrcoef(Data.loc[:,col],YTrain)[0,1]>=0,1,0)}, index=[0])
#     print(np.corrcoef(Data.loc[:,col],YTrain)[0,1])
#     NumericVariablesOrder = pd.concat([NumericVariablesOrder,Row])
NumericVariablesOrder = None

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
#            0
# count 398.00
# mean   -0.00
# std     2.12
# min    -4.13
# 25%    -1.86
# 50%     0.37
# 75%     1.60
# max     3.95
## 3) Model ----------------------------------------------------------------------------------
## Creating the Data 
Data = pd.DataFrame(XTrain)
Data['priditScore'] = priditScore
Data['priditScore'].describe()

## Creating the new unsupercised Y and adding the Actual Y
Data['NewHistoricalYTarget'] = np.where(Data['priditScore'] >= Data['priditScore'].quantile(0.10),0,1)
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

RunModel.get_modelMetricsEstimation_from_pkl(conf['Path'])
#{'Accuracy': 0.975, 'F1': 0.972972972972973, 'Precision': 1.0, 'Recall': 0.9473684210526315, 'RocCurve': (array([0., 0., 1.]), array([0.        , 0.94736842, 1.        ]), array([2, 1, 0]))}

## Checking the Model
Output['Y'] = pd.DataFrame(YTrain)
Output.groupby('Target')['PredictGBM'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)
#    Target  PredictGBM
# 0       0        0.00
# 1       1        1.00
Output.groupby('Y')['PredictGBM'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)
#    Y  PredictGBM
# 0  0        0.97
# 1  1        0.15
Output.groupby('Rank')['Y'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)
#    Rank    Y
# 0     1 1.00
# 1     2 0.97
# 2     3 1.00
# 3     4 0.97
# 4     5 0.95
# 5     6 0.85
# 6     7 0.44
# 7     8 0.07
# 8     9 0.00
# 9    10 0.00
Output.groupby('Rank')['PredictGBM'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)
#    Rank  PredictGBM
# 0     1        0.00
# 1     2        0.00
# 2     3        0.00
# 3     4        0.00
# 4     5        0.00
# 5     6        0.98
# 6     7        1.00
# 7     8        1.00
# 8     9        1.00
# 9    10        1.00

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
# 0        0        0.92
# 1        1        0.22
Predictions.groupby('Rank')['ActualY'].apply(np.mean).reset_index()
#    Rank  ActualY
# 0     1     1.00
# 1     2     1.00
# 2     3     1.00
# 3     4     1.00
# 4     5     0.69
# 5     6     0.82
# 6     7     0.67
# 7     8     0.13
# 8     9     0.00
# 9    10     0.00
Predictions.groupby('Rank')['PredictGBM'].apply(np.mean).reset_index()
#    Rank  PredictGBM
# 0     1        0.00
# 1     2        0.00
# 2     3        0.00
# 3     4        0.00
# 4     5        0.03
# 5     6        1.00
# 6     7        1.00
# 7     8        1.00
# 8     9        1.00
# 9    10        1.00

Predictions.groupby('Rank')['ActualY'].apply(np.mean).reset_index()
#    Rank  ActualY
# 0     1     1.00
# 1     2     1.00
# 2     3     1.00
# 3     4     1.00
# 4     5     0.71
# 5     6     0.88
# 6     7     0.59
# 7     8     0.18
# 8     9     0.00
# 9    10     0.00





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
#InputData = pd.read_csv("/Users/dhhazanov/UmAI/Data/Train_Inpatientdata-1542865627584.csv")
InputData = pd.read_csv("/Users/dhhazanov/UmAI/Data/insurance_claims.csv")
InputData = pd.DataFrame(InputData)
InputData = InputData.drop(['Unnamed: 0','Column1'], axis = 1,errors = 'ignore') 
#dataHealth = dataHealth.loc[:,['fraud_reported','auto_model','auto_make','police_report_available','property_damage','incident_location','incident_city','incident_state','authorities_contacted','policy_state','policy_csl','insured_sex','insured_education_level','insured_occupation','insured_hobbies','insured_relationship','incident_type','collision_type','incident_severity']]



## Create the X data and the Y data ---------------------------------
#makeClassificationY = np.where(InputData['Provider']=='PRV55912', 1, 0)
makeClassificationY = np.where(InputData['fraud_reported']=='Y', 1, 0)
makeClassificationX = InputData

XTrain, XTest, YTrain, YTest = train_test_split(InputData, makeClassificationY, test_size=0.3, random_state=0)
#pd.DataFrame(YTrain).describe()

## 2) Pridit Score ----------------------------------------------------------------------------------
## Creating the Data
Data = pd.DataFrame(XTrain).reset_index(drop = True)
Data.columns = Data.columns.astype(str)

NumericVariablesOrder = None

## Creating the configuration
conf = {
    'UsingFactor': 'Both',  ##Both, OnlyVariables, 
    'FactorVariables': None,  ##List, None
    #'NumericVariables': NumericVariables,  ##list, None
    #'FactorVariables': [],  ##List, None
    #'NumericVariables': [],  ##list, None
    'FactorsVariablesOrder': None,  ##List, None
    'NumericVariablesOrder': None  ##List, None
}
# {'UsingFactor': 'Both', 
# 'FactorVariables': ['BeneID', 'ClaimID', 'Provider', 'AttendingPhysician', 'OperatingPhysician', 'OtherPhysician', 
# 'ClmAdmitDiagnosisCode', 'DiagnosisGroupCode', 'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4', 
# 'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9', 
# 'ClmDiagnosisCode_10', 'DeductibleAmtPaid'], 
# 'NumericVariables': ['ClmProcedureCode_1', 'ClmProcedureCode_2', 'ClmProcedureCode_3', 'InscClaimAmtReimbursed', 'ClmProcedureCode_4', 'ClmProcedureCode_5', 'ClmProcedureCode_6'], 
# 'FactorsVariablesOrder':          

## Pridit
PC = PriditClassifier(Data = Data, conf = conf,logger = logger)
#PC = PriditClassifier(Data = Data, conf = {},logger = logger)
priditScore,F,firstEigenVector  = PC.Pridit()
pd.DataFrame(firstEigenVector).describe()
#           0
# count 39.00
# mean  -0.01
# std    0.16
# min   -0.42
# 25%   -0.03
# 50%    0.00
# 75%    0.03
# max    0.29
## 3) Model ----------------------------------------------------------------------------------
## Creating the Data 
Data = pd.DataFrame(XTrain).reset_index(drop = True)
Data['priditScore'] = priditScore
Data['priditScore'].describe()

## Creating the new unsupercised Y and adding the Actual Y
Data['NewHistoricalYTarget'] = np.where(Data['priditScore'] >= Data['priditScore'].quantile(0.10),0,1)
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

RunModel.get_modelMetricsEstimation_from_pkl(conf['Path'])
#{'Accuracy': 0.563514467184192, 'F1': 0.6310766477781091, 'Precision': 0.5501820072802912, 'Recall': 0.7398601398601399, 'RocCurve': (array([0.        , 0.61609687, 1.        ]), array([0.        , 0.73986014, 1.        ]), array([2, 1, 0]))}

## Checking the Model
Output['Y'] = pd.DataFrame(YTrain)
Output.groupby('Target')['PredictGBM'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)
#    Target  PredictGBM
# 0       0        0.04
# 1       1        0.97

Output.groupby('Y')['PredictGBM'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)
#    Y  PredictGBM
# 0  0        0.58
# 1  1        0.60

Output.groupby('Rank')['Y'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)
#    Rank    Y
# 0     2 0.40
# 1     3 0.42
# 2     4 0.44
# 3     5 0.37
# 4     6 0.40
# 5     7 0.48
# 6     8 0.47
# 7     9 0.55
# 8    10 0.33

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
NewData = pd.DataFrame(XTest).reset_index(drop=True)
NewData['Y'] = YTest

conf={
    'Path':'/Users/dhhazanov/UmAI/Models/Model.pckl'##Where is the model saved
}
PredictionsClass = Predict(NewData,conf,logger)
Predictions = PredictionsClass.Predict()
Predictions.describe()

Predictions['ActualY'] = NewData['Y'] 

Predictions.groupby('ActualY')['PredictGBM'].apply(np.mean).reset_index()
#    ActualY  PredictGBM
# 0        0        0.51
# 1        1        0.61
Predictions.groupby('Rank')['ActualY'].apply(np.mean).reset_index()
#    Rank  ActualY
# 0     1     0.14
# 1     2     0.12
# 2     3     0.23
# 3     4     0.29
# 4     5     0.34
# 5     6     0.10
# 6     7     0.31
# 7     8     0.43
# 8     9     0.34
# 9    10     0.24
Predictions.groupby('Rank')['PredictGBM'].apply(np.mean).reset_index()
#    Rank  PredictGBM
# 0     1        0.00
# 1     2        0.02
# 2     3        0.08
# 3     4        0.39
# 4     5        0.78
# 5     6        0.93
# 6     7        0.97
# 7     8        0.99
# 8     9        0.99
# 9    10        1.00

Predictions.groupby('Rank')['ActualY'].apply(np.mean).reset_index()
#    Rank  ActualY
# 0     1     0.14
# 1     2     0.12
# 2     3     0.23
# 3     4     0.29
# 4     5     0.34
# 5     6     0.10
# 6     7     0.31
# 7     8     0.43
# 8     9     0.34
# 9    10     0.24