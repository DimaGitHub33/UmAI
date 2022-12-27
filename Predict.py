
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

class Predict():
    ## Ranks Dictionary ----------------------------------------------------------
    def __init__(self, Data, conf, logger):
        self.Data = Data
        self.conf = conf
        self.logger = logger

        self.logger.debug('Predict was created')


    ## ------------------------------------------------------------------------------
    ## ----------------------- load_model function-----------------------------------
    ## ------------------------------------------------------------------------------
    """    
    load_model
        Args:
          self:
            All the configuration of the prediction like where are the models
            Example:
                    conf={
                         'Path':'/Users/dhhazanov/UmAI/Models/Model.pckl' ##Where is the model saved
                         }
    
          Path:
            Where to write the conf dictionary
            Example:
                    '/Users/dhhazanov/UmAI/Models/conf2'
        Returns:
          write conf in Path that inputed as json file
    
    """
    def load_model(self,Path):
        Data = self.Data
        conf = self.conf
        logger = self.logger

        ### Save The conf ----------------------------------------------------- 
        with open(Path, 'w') as convert_file:
            convert_file.write(json.dumps(conf))

        logger.debug('conf saved in = {path} '.format(path=Path))


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

        ## convert columns names to string -----------------------------------------
        Data.columns = Data.columns.astype(str)
        
        ## Reset Index to Data -----------------------------------------------------
        #Data = Data.reset_index(drop=True)

        ### Load The Models ------------------------------------------------------- 
        Path = conf['Path']

        logger.debug('using path from conf , path={path} '.format(path=Path))
        # Path = Path.replace('Segment', Segment, 1)
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
         maxY,
         minY,
         logisticRegressionModel,
         predictionsDictionary,
         CreateModelDate,
         NameColumnsOfDataInModel,
         conf] = obj

        del f, Path, obj    

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
        PredictGBM  Rank  PredictLogisticRegression VariableImportance1 VariableImportance2 VariableImportance3 VariableImportance4 VariableImportance5 VariableImportance6 VariableImportance7
             0.00     2                       0.00                [11]                [13]                 [9]                [10]                 [1]                 [7]                 [6]
             0.00     4                       0.00                [11]                [13]                 [9]                 [3]                 [2]                 [5]                 [1]
             0.00     1                       0.00                [11]                [13]                 [9]                [10]                 [0]                 [3]                 [8]
             0.00     3                       0.00                [11]                [13]                [10]                 [1]                 [8]                 [3]                [12]
             0.00     1                       0.00                [11]                [13]                 [9]                [10]                [14]                 [1]                 [6]
              ...   ...                        ...                 ...                 ...                 ...                 ...                 ...                 ...                 ...
             1.00     6                       0.99                [11]                [13]                [10]                 [9]                 [0]                 [8]                 [2]
             0.00     3                       0.00                [11]                [13]                [10]                 [0]                 [1]                 [5]                 [2]
             1.00    10                       1.00                [11]                [13]                 [9]                 [3]                 [5]                 [6]                [12]
             0.00     2                       0.00                [11]                [13]                 [9]                [10]                 [1]                 [3]                 [5]
             0.00     4                       0.01                [11]                 [9]                [13]                [10]                [12]                 [1]                 [0]
    """
    def Predict(self):
        Data = self.Data
        conf = self.conf
        logger = self.logger

        ## convert columns names to string -----------------------------------------
        Data.columns = Data.columns.astype(str)
        
        ## Reset Index to Data -----------------------------------------------------
        Data = Data.reset_index(drop=True)

        logger.debug('fit called with parameters conf={conf} '.format(conf = conf))
        
        ### Load The Models ------------------------------------------------------- 
        Path = conf['Path']

        logger.debug('using path from conf , path={path} '.format(path=Path))
        # Path = Path.replace('Segment', Segment, 1)
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
         maxY,
         minY,
         logisticRegressionModel,
         predictionsDictionary,
         CreateModelDate,
         NameColumnsOfDataInModel,
         conf] = obj
       
        del f, Path, obj   

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


        ### Predict -------------------------------------------------
        ## GBM ------------------------------------------------------
        XTest = Data.loc[:, GBMModel.feature_names].astype(float)
        ##Data['PredictGBM'] = GBMModel.predict(XTest)for GBM regressor
        Data['PredictGBM'] = GBMModel.predict_proba(XTest)[:,1]
        Data['PredictGBM'] = Data['PredictGBM'].astype(float)
        Data['PredictGBM'] = np.where(Data['PredictGBM'] >= maxY, maxY, Data['PredictGBM'])
        Data['PredictGBM'] = np.where(Data['PredictGBM'] <= minY, minY, Data['PredictGBM'])

        ## Ranks ----------------------------------------------------
        # Convert Each prediction value to rank
        Data['Rank'] = predictionsDictionary.loc[Data['PredictGBM']]['rank'].reset_index(drop=True)

        ## Logistic Regression ------------------------------------
        Data['PredictLogisticRegression'] = logisticRegressionModel.predict_proba(XTest.astype(float))[:,1]

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
                                    'PredictLogisticRegression': Data['PredictLogisticRegression'],
                                    'VariableImportance1': Data['VariableImportance1'],
                                    'VariableImportance2': Data['VariableImportance2'],
                                    'VariableImportance3': Data['VariableImportance3'],
                                    'VariableImportance4': Data['VariableImportance4'],
                                    'VariableImportance5': Data['VariableImportance5'],
                                    'VariableImportance6': Data['VariableImportance6'],
                                    'VariableImportance7': Data['VariableImportance7']})

        return Output

#Check The Model --------------------------------------------------------  
# from sklearn.datasets import make_classification
# makeClassificationX,makeClassificationY = make_classification(n_samples=10000,class_sep = 4,random_state=0)
# Data = pd.DataFrame(makeClassificationX)
# Data['Y'] = pd.DataFrame(makeClassificationY)

# conf={
#     'Path':'/Users/dhhazanov/UmAI/Models/Model.pckl'
# }
# Predictions = Predict(Data,conf,logger).Predict()
# Predictions.describe()

# Predictions['ActualY'] = pd.DataFrame(makeClassificationY)
# Predictions.groupby('ActualY')['PredictGBM'].apply(np.mean).reset_index()
# Predictions.groupby('Rank')['PredictGBM'].apply(np.mean).reset_index()
# Predictions.groupby('Rank')['ActualY'].apply(np.mean).reset_index()
# Predictions.groupby('Rank')['PredictLogisticRegression'].apply(np.mean).reset_index()


#Check The Model 2 --------------------------------------------------------  
# Data = pd.read_csv('/Users/dhhazanov/UmAI/Eli_data_health.csv')
# conf={
#     'Path':'/Users/dhhazanov/UmAI/Models/Model.pckl'
# }
# Predictions = Predict(Data,conf,logger).Predict()
# Predictions.describe()

# Predictions['ActualY'] = np.where(Data['GIL'] >= Data['GIL'].mean(),1,0)
# Predictions.groupby('ActualY')['PredictGBM'].apply(np.mean).reset_index()
# Predictions.groupby('Rank')['PredictGBM'].apply(np.mean).reset_index()
# Predictions.groupby('Rank')['ActualY'].apply(np.mean).reset_index()
# Predictions.groupby('Rank')['PredictLogisticRegression'].apply(np.mean).reset_index()


#Check The Model 3 --------------------------------------------------------  
# Data = pd.read_csv('/Users/dhhazanov/UmAI/ppp.parquet_1_test_for_predict.csv')
# Data['Y'] = np.where(Data['GIL'] >= Data['GIL'].mean(),1,0)
# conf={
#     'Path':'/Users/dhhazanov/UmAI/Models/Model.pckl'
# }
# import re
# Data = Data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))  
# Predictions = Predict(Data,conf,logger).Predict()
# Predictions.describe()   