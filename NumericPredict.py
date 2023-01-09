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
import logging as logger
import json
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
## ----------------------------------------- Predict ----------------------------------------------
## ------------------------------------------------------------------------------------------------

class NumericPredict():
    ## Ranks Dictionary ----------------------------------------------------------
    def __init__(self, Data, conf, logger):
        self.Data = Data
        self.conf = conf
        self.logger = logger

        self.logger.debug('Predict was created')


      ## ------------------------------------------------------------------------------
    ## ----------------------------- read_pickle  -----------------------------------
    ## ------------------------------------------------------------------------------
    """    
    read_pickle
        Args:
        Returns:
          Model pickle
    """
    def read_pickle(self,path):
        
        logger = self.logger
        logger.debug('Read the pickle from {path} '.format(path = path))

        ## Read the pickle
        f = open(path, 'rb')
        obj = pickle.load(f)
        f.close()     

        return obj

    ## ------------------------------------------------------------------------------
    ## ----------------------- get_conf_from_pkl  -----------------------------------
    ## ------------------------------------------------------------------------------
    """    
    load_model
        Args:
          Path:
            Where the model pickle was saved
            Example:
                    '/Users/dhhazanov/UmAI/Models/conf2'
        Returns:
          conf model
    
    """
    def get_conf_from_pkl(self,path):
        Data = self.Data
        conf = self.conf
        logger = self.logger
        read_pickle = self.read_pickle

        logger.debug('get conf from pkl from {path} '.format(path = path))

        ### Load The pickle ------------------------------------------------------- 
        [factorVariables,
         numericVariables,
         YMCFactorDictionaryList,
         totalYMeanTarget,
         totalYMedianTarget,
         YMCDictionaryNumericList,
         GBMModel, maxY, minY,
         predictionsDictionary,
         QrModel, UpperBorder, UpperValue, Calibration,
         CreateModelDate,
         NameColumnsOfDataInModel,
         conf,
         Mappe]= read_pickle(path)

        return conf


    ## ------------------------------------------------------------------------------
    ## ------------------- pre_predict_validation function --------------------------
    ## ------------------------------------------------------------------------------
    """    
    pre_predict_validation
        Args:
        Returns:
          1) Flag (True/False) if the predicted data are all in the trained model data
          2) What columns in the predicted data are not in the trained model data  
    """
    def pre_predict_validation(self):
        Data = self.Data
        conf = self.conf
        logger = self.logger
        read_pickle = self.read_pickle


        ### Load The pickle ------------------------------------------------------- 
        [factorVariables,
        numericVariables,
        YMCFactorDictionaryList,
        totalYMeanTarget,
        totalYMedianTarget,
        YMCDictionaryNumericList,
        GBMModel, maxY, minY,
        predictionsDictionary,
        QrModel, UpperBorder, UpperValue, Calibration,
        CreateModelDate,
        NameColumnsOfDataInModel,
        conf,
        Mappe] = read_pickle(path = conf['Path'])

        ## convert columns names to string -----------------------------------------
        Data.columns = Data.columns.astype(str)

        ### Check if existed in the predicted data the columns in the trained model data --
        Flag = set(NameColumnsOfDataInModel).issubset(set(Data.columns))##set([1,2,3]).issubset(set([1,2,3,4])) return TRUE
        Difference = list(set(NameColumnsOfDataInModel).difference(Data.columns))##set([1,2,3,4]).difference([1,2,3]) return 4

        if (Flag == True):
            logger.debug('All the trained models data columns are in the predicted data')
        else:
            logger.debug('{Difference} Not in the trained model data but in the predicted'.format(Difference = Difference))


        return Flag,Difference
        
    ## ------------------------------------------------------------------------------
    ## ---------------------------- Predict function --------------------------------
    ## ------------------------------------------------------------------------------
    """    
    Predict
        Args:
        Returns:
        1) PredictGBM  
        2) Rank  
        3) PredictLogisticRegression 
        4) VariableImportance1 
        5) VariableImportance2 
        6) VariableImportance3 
        7) VariableImportance4 
        8) VariableImportance5 
        9) VariableImportance6 
        10) VariableImportance7

        Example:
                PredictGBM  Rank  PredictQuantile  PredictCalibrated VariableImportance1 VariableImportance2 VariableImportance3 VariableImportance4 VariableImportance5 VariableImportance6 VariableImportance7
                    95.02      7           117.10             118.58                 [5]                 [6]                 [4]                 [2]                 [9]                 [8]                 [3]
                    273.12    10           270.39             272.20                 [9]                 [6]                 [4]                 [2]                 [8]                 [1]                 [5]
                    -189.48    2          -181.02            -181.82                 [2]                 [9]                 [5]                 [8]                 [4]                 [3]                 [1]
                    229.24     9           191.21             190.79                 [6]                 [4]                 [9]                 [7]                 [5]                 [1]                 [8]
                    172.63     9           152.47             152.52                 [4]                 [5]                 [2]                 [6]                 [3]                 [1]                 [8]
                    ...      ...              ...                ...                 ...                 ...                 ...                 ...                 ...                 ...                 ...
                    60.17      7            -1.43              -1.74                 [5]                 [3]                 [6]                 [1]                 [9]                 [2]                 [8]
                    210.17     9           226.55             226.49                 [9]                 [8]                 [4]                 [2]                 [5]                 [1]                 [6]
                    144.20     8           149.64             149.69                 [2]                 [3]                 [7]                 [4]                 [6]                 [9]                 [8]
                    67.96      7            34.38              32.13                 [9]                 [6]                 [2]                 [8]                 [7]                 [5]                 [0]
                    -69.11     4           -65.02             -64.74                 [5]                 [9]                 [6]                 [8]                 [4]                 [3]                 [1]
    """
    def Predict(self):
        Data = self.Data
        conf = self.conf
        logger = self.logger
        read_pickle = self.read_pickle

        ## convert columns names to string -----------------------------------------
        Data.columns = Data.columns.astype(str)
        
        ## Reset Index to Data -----------------------------------------------------
        Data = Data.reset_index(drop=True)

        logger.debug('fit called with parameters conf={conf} '.format(conf = conf))
        
        ### Load The pickle ------------------------------------------------------- 
        [factorVariables,
        numericVariables,
        YMCFactorDictionaryList,
        totalYMeanTarget,
        totalYMedianTarget,
        YMCDictionaryNumericList,
        GBMModel, maxY, minY,
        predictionsDictionary,
        QrModel, UpperBorder, UpperValue, Calibration,
        CreateModelDate,
        NameColumnsOfDataInModel,
        conf,
        Mappe] = read_pickle(path = conf['Path'])

        ### Inserting the YMC Values from the dictionaries to the DataPanel -------
        for variableName in YMCFactorDictionaryList:
            # print(variableName)
            # variableName="M_ERUAS"
            Data.loc[:, variableName] = Data[variableName].astype(str)

            YMCDictionary = YMCFactorDictionaryList[variableName]
            YMCDictionary.columns = [variableName,
                                     variableName + "_MeanFactorYMC",
                                     variableName + "_MedianFactorYMC"]

            Data = Data.join(YMCDictionary.set_index([variableName]), how='left', on=[variableName])
            Data.loc[:, variableName + "_MeanFactorYMC"] = Data[variableName + "_MeanFactorYMC"].fillna(totalYMeanTarget)
            Data.loc[:, variableName + "_MedianFactorYMC"] = Data[variableName + "_MedianFactorYMC"].fillna(totalYMedianTarget)

            ### Delete all temporary Variables ----------------------------------------
            del YMCDictionary
            del variableName 
            
        ### Creating the YMC calculation for each numeric variable ----------------
        numericYMC = pd.DataFrame(data={})
        for variableToConvert in YMCDictionaryNumericList:
            # print(variableToConvert)
            Variable = pd.DataFrame(data={variableToConvert: Data[variableToConvert].astype(float)})
            Variable.loc[:, variableToConvert] = Variable[variableToConvert].fillna(0)

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
        # Data = Data.join(Numeric_YMC.set_index('ClaimNo_Descision'), how='left', on='ClaimNo_Descision')
        Data = pd.concat([Data, numericYMC], axis=1)

        ### Delete all temporary Variables ----------------------------------------
        del numericYMC

        ## ------------------------------------------------------------------------
        ### GBM Predict -----------------------------------------------------------
        ## ------------------------------------------------------------------------
        XTest = Data.loc[:, GBMModel.feature_names].astype(float)
        Data['PredictGBM'] = GBMModel.predict(XTest)##for GBM regressor
        ##Data['PredictGBM'] = GBMModel.predict_proba(XTest)[:,1]
        Data['PredictGBM'] = Data['PredictGBM'].astype(float)
        Data['PredictGBM'] = np.where(Data['PredictGBM'] >= maxY, maxY, Data['PredictGBM'])
        Data['PredictGBM'] = np.where(Data['PredictGBM'] <= minY, minY, Data['PredictGBM'])

        ## Ranks ------------------------------------------------------------------
        # Convert Each prediction value to rank
        Data['Rank'] = predictionsDictionary.loc[Data['PredictGBM']]['rank'].reset_index(drop=True)


        ## ------------------------------------------------------------------------
        ### Calibration Predict -------------------------------------------------
        ## ------------------------------------------------------------------------
        Data['PredictQuantile'] = QrModel.predict(XTest)
        #Data['PredictQuantile'] = np.maximum(1,Data['PredictQuantile'])
        
        ##LTV Upper Cut Point
        Data['PredictQuantile'] = np.where(Data['PredictQuantile']>=UpperBorder, UpperValue, Data['PredictQuantile'])
           
        # Calibration ----------------------------------------------------------
        Data['Calibration'] = Calibration.loc[Data['PredictQuantile']]['Calibration'].reset_index(drop=True)
        Data['PredictCalibrated'] = Data['Calibration'] * Data['PredictQuantile']


        ### Variable Explainer - ----------------------------------
        explainer = shap.Explainer(GBMModel, Data.loc[:, GBMModel.feature_names].astype(float), feature_names=GBMModel.feature_names)  # Instead of Explainer put TreeExplainer if the model is tree based
        shap_values = explainer(Data.loc[:, GBMModel.feature_names].astype(float))
        # shap.summary_plot(shap_values, Data.loc[:,GBMModel.feature_names].astype(float))
        # shap.summary_plot(shap_values, Data.loc[:,GBMModel.feature_names].astype(float), plot_type='bar')
        shap_values_frame = pd.DataFrame(shap_values.values).abs()
        shap_values_frame.columns = shap_values.feature_names
        # shap_values_frame = shap_values_frame.loc[Out.index,:]

        AllVariableImportance = pd.DataFrame()
        for i in shap_values_frame.index:
            svf = pd.DataFrame(shap_values_frame.loc[i, :])
            svf.columns = ['Importance']
            svf = svf.sort_values(by='Importance', ascending=False).head(100)
            svf = svf.reset_index()
            svf.columns = ['Variable', 'Importance']
            svf['VariableTransformation'] = list(map(lambda x: x.replace('_MeanNumericYMC', '').replace('_MedianFactorYMC', '').replace('_Quantile99FactorYMC', '').replace('_SdFactorYMC', '').replace('_Numeric', '').replace('_MeanFactorYMC', '').replace('_MedianFactorYMC', ''), svf['Variable']))
            svf = svf.loc[~svf.VariableTransformation.duplicated(), :]
            Variables = svf['VariableTransformation'].reset_index(drop=True)
            Variables = pd.concat([Variables, pd.DataFrame({'None1', 'None2', 'None3', 'None4', 'None5', 'None6'}).reset_index(drop=True)], axis=0).values
            VariableImportance = pd.DataFrame(data={'Index': [i],
                                                    'VariableImportance1': [Variables[0]],
                                                    'VariableImportance2': [Variables[1]],
                                                    'VariableImportance3': [Variables[2]],
                                                    'VariableImportance4': [Variables[3]],
                                                    'VariableImportance5': [Variables[4]],
                                                    'VariableImportance6': [Variables[5]],
                                                    'VariableImportance7': [Variables[6]]
                                                    })
            AllVariableImportance = pd.concat([AllVariableImportance, VariableImportance])

        AllVariableImportance = AllVariableImportance.set_index('Index')
        Data = pd.concat([Data.reset_index(drop=True), AllVariableImportance], axis=1)

        ### Output ----------------------------------------------------------------
        Output = pd.DataFrame(data={'PredictGBM': Data['PredictGBM'],
                                    'Rank': Data['Rank'],
                                    'PredictQuantile': Data['PredictQuantile'],
                                    'PredictCalibrated': Data['PredictCalibrated'],
                                    'VariableImportance1': Data['VariableImportance1'],
                                    'VariableImportance2': Data['VariableImportance2'],
                                    'VariableImportance3': Data['VariableImportance3'],
                                    'VariableImportance4': Data['VariableImportance4'],
                                    'VariableImportance5': Data['VariableImportance5'],
                                    'VariableImportance6': Data['VariableImportance6'],
                                    'VariableImportance7': Data['VariableImportance7']})

        return Output

# #Check The Model --------------------------------------------------------  
# Data = pd.read_csv('/Users/dhhazanov/UmAI/Data/insurance_claims.csv')
# conf={
#      'Path':'/Users/dhhazanov/UmAI/Models/Model.pckl'
# }
# import re
# Data = Data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))  
# Predictions = Predict(Data,conf,logger).Predict()
# Predictions.describe()   