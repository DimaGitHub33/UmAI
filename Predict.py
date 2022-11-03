
import os
import pickle
from datetime import datetime
from docutils import DataError

import lightgbm as LightGBM
import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV

from Functions import Ranks_Dictionary, RJitter, FunFactorYMC, FunNumericYMC

#from sklearn.linear_model import LogisticRegression
#import lifelines
#from lifelines.utils import k_fold_cross_validation

#import pyarrow.parquet as pq
#from sklearn.linear_model import QuantileRegressor
#import shap
#import sqlalchemy
#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.model_selection import GridSearchCV
#import statsmodels.formula.api as smf
#from sklearn import metrics


## ------------------------------------------------------------------------------------------------
## ----------------------------------------- Model ------------------------------------------------
## ------------------------------------------------------------------------------------------------

class Predict():
    ## Ranks Dictionary ----------------------------------------------------------
    def __init__(self, Data, conf):
        self.Data = Data
        self.conf = conf

    def Predict(self):
        Data = self.Data
        conf = self.conf

        ### Load The Models ------------------------------------------------------- 
        Path = conf['Path']  
        #Path = Path.replace('Segment', Segment, 1)
        f = open(Path, 'rb')
        obj = pickle.load(f)
        f.close()
        
        [factorVariables,
         numericVariables,
         YMCFactorDictionaryList,
         totalYMeanTarget,
         totalYMedianTarget,
         YMCDictionaryNumericList,
         GBMModel,
         CreateModelDate] = obj
       
        del f, Path, obj   
                

        ### Inserting the YMC Values from the dictionaries to the DataPanel -------
        for variableName in YMCFactorDictionaryList:
            #print(variableName)
            # variableName="M_ERUAS"
            Data.loc[:, variableName] = Data[variableName].astype(str)
            
            YMCDictionary = YMCFactorDictionaryList[variableName]
            YMCDictionary.columns = [variableName, 
                                     variableName+"_MeanFactorYMC",
                                     variableName+"_MedianFactorYMC"]
        
            Data = Data.join(YMCDictionary.set_index([variableName]), how='left', on=[variableName])
            Data.loc[:, variableName+"_MeanFactorYMC"] = Data[variableName+"_MeanFactorYMC"].fillna(totalYMeanTarget)        
            Data.loc[:, variableName+"_MedianFactorYMC"] = Data[variableName+"_MedianFactorYMC"].fillna(totalYMedianTarget)
        
        ### Delete all temporary Variables ----------------------------------------
        del YMCDictionary
        del variableName 
            
        ### Creating the YMC calculation for each numeric variable ----------------
        numericYMC = pd.DataFrame(data={})
        for variableToConvert in YMCDictionaryNumericList:
            #print(variableToConvert)
            Variable = pd.DataFrame(data={variableToConvert: Data[variableToConvert].astype(float)})
            Variable.loc[:,variableToConvert] = Variable[variableToConvert].fillna(0)
        
            # Inserting the numeric dictionary into VariableDictionary
            variableDictionary = YMCDictionaryNumericList[variableToConvert]
        
            # Adding All the YMC
            variableDictionary.index = pd.IntervalIndex.from_arrays(variableDictionary['lag_value'],
                                                                    variableDictionary['value'],
                                                                    closed='left')
            V = Variable[variableToConvert]
            Variable[['MeanNumericYMC']] = variableDictionary.loc[V][['MeanNumericYMC']].reset_index(drop=True)
        
            # Creating YMC table
            YMC = pd.DataFrame(data={'variableToConvert_MeanNumericYMC': Variable['MeanNumericYMC']})
            
            # Left join YMC table to NUmeric_YMC table
            numericYMC = pd.concat([numericYMC, YMC], axis=1)
            numericYMC.columns = list(map(lambda x: x.replace('variableToConvert', variableToConvert, 1), numericYMC.columns))
        
        ### Left join of Numeric_YMC table to the DataPanel -----------------------
        #Data = Data.join(Numeric_YMC.set_index('ClaimNo_Descision'), how='left', on='ClaimNo_Descision')
        Data = pd.concat([Data,numericYMC], axis = 1)
        
        ### Delete all temporary Variables ----------------------------------------
        del variableToConvert
        del numericYMC 
        del Variable
        del variableDictionary
        del V
        del YMC
        
        
        ### -----------------------------------------------------------------------
        ### ----------------------- Target Model ----------------------------------
        ### -----------------------------------------------------------------------
        ### Taking the YMC_Suspicious variables -----------------------------------

        XTest = Data.loc[:,GBMModel.feature_names].astype(float)
        Data['predictGBM'] = GBMModel.predict(XTest)
        Data['predictGBM'] = Data['predictGBM'].astype(float)

        ### Output ----------------------------------------------------------------
        Output = pd.DataFrame(data={'predictGBM': Data['predictGBM']})

        return Output


#Check The Model --------------------------------------------------------  
# Data = pd.read_parquet('/Users/dhhazanov/Downloads/ppp_v1.parquet.gzip', engine='pyarrow')
# conf={
#     'Path':'/Users/dhhazanov/UmAI/Models/Model.pckl'
# }
# Predictions = Predict(Data,conf).Predict()
# Predictions.describe()

# Data['Predictions'] = Predictions

# Data['Target'] = np.where(Data['GIL'] >= Data['GIL'].mean(),1,0)
# AggregationTable = Data.groupby('Target')['Predictions'].apply(np.mean).reset_index()
# pd.options.display.float_format = '{:.2f}'.format