
import os
import pickle
from datetime import datetime
from docutils import DataError

import lightgbm as LightGBM
import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV

from Functions import Ranks_Dictionary, RJitter, FunFactorYMC, FunNumericYMC
pd.options.display.float_format = '{:.2f}'.format

import shap
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
        
            ### Delete all temporary Variables ----------------------------------------
            del variableToConvert
            del Variable
            del variableDictionary
            del V
            del YMC

        ### Left join of Numeric_YMC table to the DataPanel -----------------------
        #Data = Data.join(Numeric_YMC.set_index('ClaimNo_Descision'), how='left', on='ClaimNo_Descision')
        Data = pd.concat([Data,numericYMC], axis = 1)
        
        ### Delete all temporary Variables ----------------------------------------
        del numericYMC 

        
        
        ### -----------------------------------------------------------------------
        ### ----------------------- Target Model ----------------------------------
        ### -----------------------------------------------------------------------
        ### Taking the YMC_Suspicious variables -----------------------------------

        XTest = Data.loc[:,GBMModel.feature_names].astype(float)
        Data['predictGBM'] = GBMModel.predict(XTest)
        Data['predictGBM'] = Data['predictGBM'].astype(float)


        ### Variable Explainer - ----------------------------------
        explainer = shap.Explainer(GBMModel, Data.loc[:,GBMModel.feature_names].astype(float), feature_names = GBMModel.feature_names) #Instead of Explainer put TreeExplainer if the model is tree based
        shap_values = explainer(Data.loc[:,GBMModel.feature_names].astype(float))
        #shap.summary_plot(shap_values, Data.loc[:,GBMModel.feature_names].astype(float))
        #shap.summary_plot(shap_values, Data.loc[:,GBMModel.feature_names].astype(float), plot_type='bar')
        shap_values_frame = pd.DataFrame(shap_values.values).abs()
        shap_values_frame.columns = shap_values.feature_names
        #shap_values_frame = shap_values_frame.loc[Out.index,:]
        
        
        AllVariableImportance = pd.DataFrame()
        for i in shap_values_frame.index:
            svf = pd.DataFrame(shap_values_frame.loc[i,:])
            svf.columns = ['Importance']
            svf = svf.sort_values(by = 'Importance', ascending=False).head(100)
            svf = svf.reset_index()
            svf.columns = ['Variable','Importance']
            svf['VariableTransformation'] = list(map(lambda x: x.replace('_MeanNumericYMC','').replace('_MedianFactorYMC','').replace('_Quantile99FactorYMC','').replace('_SdFactorYMC','').replace('_Numeric','').replace('_MeanFactorYMC','').replace('_MedianFactorYMC',''), svf['Variable']))
            svf = svf.loc[~svf.VariableTransformation.duplicated(),:]
            Variables = svf['VariableTransformation'].reset_index(drop=True)
            Variables = pd.concat([Variables,pd.DataFrame({'None1','None2','None3','None4','None5','None6'}).reset_index(drop=True)],axis=0).values
            VariableImportance = pd.DataFrame(data={'Index': [i],
                                                    'VariableImportance1': [Variables[0]],
                                                    'VariableImportance2': [Variables[1]],
                                                    'VariableImportance3': [Variables[2]],
                                                    'VariableImportance4': [Variables[3]],
                                                    'VariableImportance5': [Variables[4]],
                                                    'VariableImportance6': [Variables[5]],
                                                    'VariableImportance7': [Variables[6]]
                                            })
            AllVariableImportance = pd.concat([AllVariableImportance,VariableImportance])
        
        AllVariableImportance = AllVariableImportance.set_index('Index')    
        Data = pd.concat([Data.reset_index(drop=True), AllVariableImportance], axis=1)
        

        ### Output ----------------------------------------------------------------
        Output = pd.DataFrame(data={'predictGBM': Data['predictGBM'],

                                    'VariableImportance1': Data['VariableImportance1'],
                                    'VariableImportance2': Data['VariableImportance2'],
                                    'VariableImportance3': Data['VariableImportance3'],
                                    'VariableImportance4': Data['VariableImportance4'],
                                    'VariableImportance5': Data['VariableImportance5'],
                                    'VariableImportance6': Data['VariableImportance6'],
                                    'VariableImportance7': Data['VariableImportance7']})

        return Output


##Check The Model --------------------------------------------------------  
# Data = pd.read_parquet('/Users/dhhazanov/Downloads/ppp_v1.parquet.gzip', engine='pyarrow')
# conf={
#     'Path':'/Users/dhhazanov/UmAI/Models/Model.pckl'
# }
# Predictions = Predict(Data,conf).Predict()
# Predictions.describe()

# Data['Predictions'] = Predictions['predictGBM']

# Data['Target'] = np.where(Data['GIL'] >= Data['GIL'].mean(),1,0)
# AggregationTable = Data.groupby('Target')['Predictions'].apply(np.mean).reset_index()
# AggregationTable