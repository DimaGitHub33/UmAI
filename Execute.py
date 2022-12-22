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
PredictClass.load_model(Path = '/Users/dhhazanov/UmAI/Models/conf2')##Path is where to write the configuration
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
dataHealth = pd.read_csv("/Users/dhhazanov/Downloads/Eli_data_health.csv")
dataHealth = pd.DataFrame(dataHealth)
dataHealth = dataHealth.drop(['Unnamed: 0','Column1'], axis = 1)


# makeClassificationX = dataHealth.drop('GIL',axis = 1)
# makeClassificationY = np.where(dataHealth['GIL'] >= dataHealth['GIL'].mean(),1,0) 
makeClassificationX = dataHealth.drop('HAVE_HAKIRA',axis = 1)
makeClassificationY = dataHealth['HAVE_HAKIRA'].fillna(0)

XTrain, XTest, YTrain, YTest = train_test_split(makeClassificationX, makeClassificationY, test_size=0.3, random_state=0)
pd.DataFrame(YTrain).describe()


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
    # 'UsingFacotr': 'OnlyVariables',  ##Both, OnlyVariables, None
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

## Checking the Model
Output['Y'] = pd.DataFrame(YTrain)
Output.groupby('Target')['PredictGBM'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)
#    Target  PredictGBM
# 0       0        0.04
# 1       1        0.97

Output.groupby('Y')['PredictGBM'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)
#    Y  PredictGBM
# 0  0        0.49
# 1  1        0.56

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
NewData = pd.concat([pd.DataFrame(XTest),pd.DataFrame({'Y':YTest})],axis=1)

conf={
    'Path':'/Users/dhhazanov/UmAI/Models/Model.pckl'##Where is the model saved
}
Predictions = Predict(NewData,conf,logger).Predict()
Predictions.describe()

Predictions['ActualY'] = NewData['Y'] 

Predictions.groupby('ActualY')['PredictGBM'].apply(np.mean).reset_index()
#    ActualY  PredictGBM
# 0     0.00        0.48
# 1     1.00        0.55
Predictions.groupby('Rank')['ActualY'].apply(np.mean).reset_index()
#    Rank  ActualY
# 0     6     0.40
# 1     7     0.43
# 2     8     0.42
# 3     9     0.48
# 4    10     0.55
Predictions.groupby('Rank')['PredictGBM'].apply(np.mean).reset_index()
#    Rank  PredictGBM
# 0     6        0.00
# 1     7        0.10
# 2     8        0.92
# 3     9        1.00
# 4    10        1.00

Predictions.groupby('Rank')['ActualY'].apply(np.mean).reset_index()
#    Rank  ActualY
# 0     6     0.40
# 1     7     0.43
# 2     8     0.42
# 3     9     0.48
# 4    10     0.55