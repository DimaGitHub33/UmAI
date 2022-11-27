## 0) Import Packages -------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from Pridit import PriditClassifier
from Functions import Ranks_Dictionary, RJitter, FunFactorYMC, FunNumericYMC
from Model import Model
from Predict import Predict
from sklearn.datasets import load_breast_cancer
breast_cancer_x,breast_cancer_y = load_breast_cancer(return_X_y=True)

from sklearn.datasets import make_classification
make_classification_x,make_classification_y = make_classification(n_samples=1000)

## 1) Pridit Score ----------------------------------------------------------------------------------
# conf = {
#     'UsingFacotr': 'OnlyVariables',  ##Both, OnlyVariables, None
#     'FactorVariables': FactorVariables,  ##List, None
#     'NumericVariables': NumericVariables,  ##list, None
#     #'FactorVariables': [],  ##List, None
#     #'NumericVariables': [],  ##list, None
#     'FactorsVariablesOrder': None,  ##List, None
#     'NumericVariablesOrder': None  ##List, None
# }
dataHealth = pd.read_csv("/Users/dhhazanov/Downloads/Eli_data_health.csv")
PC = PriditClassifier(pd.DataFrame(dataHealth), conf={})
priditScore,F,firstEigenVector  = PC.Pridit()

## 2) Model ----------------------------------------------------------------------------------
## Creating the Data 
Data = pd.DataFrame(dataHealth)
Data['priditScore'] = priditScore
Data['priditScore'].describe()

Data['NewHistoricalYTarget'] = np.where(Data['priditScore'] >= Data['priditScore'].mean(),1,0)
conf={
    'Path':'/Users/dhhazanov/UmAI/Models/Model.pckl',
    'Target':'NewHistoricalYTarget',
    'ColumnSelectionType': 'Drop',#Drop,Keep
    'Keep': None,#['GENDER', 'FAMILY_STATUS','GIL'],
    'Drop': ['priditScore'],#['GIL','ISUK_MERAKEZ','FAMILY_STATUS','ISHUN','M_CHOD_TASHLOM_BR'],#None,
    'ModelType': None #GBM,Linear regression,...
}
RunModel = Model(Data,conf)
Output = RunModel.fit()

Output['RealY'] = pd.DataFrame(make_classification_y)
Output.groupby('Target')['predictGBM'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)
Output.groupby('RealY')['predictGBM'].apply(np.mean).reset_index()##Checking the model (Train Test is the same)


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