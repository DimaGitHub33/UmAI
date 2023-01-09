
import os
import pickle
from datetime import datetime

import lightgbm as LightGBM
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.linear_model import LogisticRegression

from Functions import Ranks_Dictionary, RJitter, FunFactorYMC, FunNumericYMC
pd.options.display.float_format = '{:.2f}'.format
import warnings
import logging as logger
import re
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

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
class Model():
    ## Ranks Dictionary -----------------------------------------------------------
    def __init__(self, Data, conf, logger):
        self.Data = Data
        self.conf = conf
        self.logger = logger

        self.logger.debug('Model was created')

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
    def pre_model_validation(self):
        Data = self.Data
        conf = self.conf
        logger = self.logger

        ## convert columns names to string -----------------------------------------
        Data.columns = Data.columns.astype(str)
        
        ## Reset Index to Data -----------------------------------------------------
        Data = Data.reset_index(drop=True)

        ### Check if the columns names are wrong -----------------------------------
        Flag = False
        correctColumnsName = Data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x)).columns
        wrongColNames = list(set(correctColumnsName) - set(Data.columns))
        
        if (len(wrongColNames) > 0):
            #raise Exception("wrong columns name ->" + str(wrongColNames))
            Flag = True

        if (Flag == True):
            wrongColNames = 'Data columns are correct'
            logger.debug(wrongColNames)
        else:
            wrongColNames = 'the columns: {wrongColNames} are not in the right format'.format(wrongColNames = wrongColNames)
            logger.debug(wrongColNames)


        return Flag,wrongColNames
    
    """    
    get_modelMetricsEstimation_from_pkl
        Args:
        Returns: Model metrics estimation 
        Example: 
                {'Accuracy': 0.9, 
                'F1': 0.9014084507042254, 
                'Precision': 0.9411764705882353, 
                'Recall': 0.8648648648648649, 
                'PrecisionRecallCurve': (
                                            array([0.52857143, 0.53623188, 0.54411765, 0.55223881, 0.56060606,
                                                   0.13513514, 0.10810811, 0.08108108, 0.05405405, 0.02702703]), 
                                            array([3.19794303e-12, 8.15585695e-12, 8.64222297e-12, 1.89704868e-11,
                                                   1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,1.00000000e+00, 1.00000000e+00])
                                        ), 
                'RocCurve': (
                             array([0.        , 0.        , 0.        , 0.03030303, 0.03030303,
                             0.06060606, 0.06060606, 0.09090909, 0.09090909, 0.12121212,0.12121212]), 
                             array([0.        , 0.02702703, 0.64864865, 0.64864865, 0.83783784,
                                    0.83783784, 0.89189189, 0.89189189, 0.91891892, 0.91891892]), 
                             array([2.00000000e+00, 1.00000000e+00, 9.99975997e-01, 9.99967249e-01,
                                    9.91333851e-01, 8.51420026e-01, 3.93920028e-01, 2.61575334e-01])
                            )
                }
    """
    def get_modelMetricsEstimation_from_pkl(self,path):
        Data = self.Data
        conf = self.conf
        logger = self.logger

        logger.debug('get modelMetricsEstimation from pkl from {path} '.format(path = path))
        
        ## Read the pickl e-------------------------------------------------------
        f = open(path, 'rb')
        obj = pickle.load(f)
        f.close()   

        ### Load The pickle ------------------------------------------------------- 
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
         conf,
         modelMetricsEstimation] = obj

        return modelMetricsEstimation
    ## ------------------------------------------------------------------------------
    ## ------------------- pre_predict_validation function --------------------------
    ## ------------------------------------------------------------------------------
    def fit(self):
        Data = self.Data
        conf = self.conf
        logger = self.logger

        logger.debug('fit called with parameters conf={conf} '.format(conf = conf))

        logger.info('check conf params ')

        ## convert columns names to string -----------------------------------------
        Data.columns = Data.columns.astype(str)
        
        ## Reset Index to Data -----------------------------------------------------
        Data = Data.reset_index(drop=True)

        ## Fill Configuration ------------------------------------------------------
        if (not 'factorVariables' in conf or conf['factorVariables'] == None):
            logger.info('factorVariables not found in conf using default -> []')

            conf['factorVariables'] = []
            factorVariables = conf['factorVariables']
        if (not 'numericVariables' in conf or conf['numericVariables'] == None):
            logger.info('numericVariables not found in conf using default -> []]')
            conf['numericVariables'] = []
            numericVariables = conf['numericVariables']
        if (not 'ColumnSelectionType' in conf or conf['ColumnSelectionType'] == None):
            logger.info('ColumnSelectionType not found in conf using default -> None')
            logger.info('ColumnSelectionType not found in conf using default for \"Keep\" -> []')
            logger.info('ColumnSelectionType not found in conf using default for \"Drop\" -> []')
            conf['ColumnSelectionType'] = []
            conf['Keep'] = []
            conf['Drop'] = []
        if (conf['ColumnSelectionType'] != None and conf['Keep'] == None):
            logger.info('ColumnSelectionType not found in conf using default for \"Keep\" -> []')
            conf['Keep'] = []
        if (not 'ValidationDataPercent' in conf or conf['ValidationDataPercent']== None):
            logger.info('ValidationDataPercent not found in conf using default -> 0.1]')
            conf['ValidationDataPercent'] = 0.1

        ## Save Target columns -----------------------------------------------------
        Target = Data[conf['Target']]

        logger.info('Target  -> {Target}'.format(Target=Target))
        try:
            logger.info('dropping target {Target}'.format(Target=Target))
            Data = Data.drop(conf['Target'], axis=1)
        except Exception:

            logger.info("Target column is not exist")

        ## Drop ot Select columns from Data ----------------------------------------
        if (len(conf['ColumnSelectionType']) != 0 and len(conf['Keep']) != 0):
            try:
                logger.info('conf[\'Keep\']={columns}'.format(columns=conf['Keep']))
                Data = Data.loc[:, np.intersect1d(Data.columns.values, conf['Keep'])]
                logger.info('Selected chosen columns , columns={columns}'.format(columns=Data.columns))
            except Exception:
                logger.info("Didn't selected anything")
                raise

        if (len(conf['ColumnSelectionType']) != 0 and len(conf['Drop']) != 0):
            try:
                if(len(np.intersect1d(Data.columns.values,conf['Drop']))==0):
                    logger.info("Didn't drop any columns")         
                else:
                    logger.info('dropping columns using , conf[\'Drop\']={drop}'.format(drop=conf['Drop']))
                    Data = Data.drop(list(np.intersect1d(Data.columns.values, conf['Drop'])), axis=1)
                    logger.info('Droped selected columns')
            except Exception:           
                logger.info("Didn't drop any columns")      
                raise  
        
        ## Insert Target columns -----------------------------------------------------
        Data['Target'] = Target
        del Target

        ## factorVariables and numericVariables Variables --------------------------
        factorVariables = conf['factorVariables']
        numericVariables = conf['numericVariables']

        ## Fill the FactorVariables and NumericVariables list ----------------------
        if factorVariables is None or len(factorVariables) == 0:
            factorVariables = []
            data_types = Data.dtypes.reset_index().rename(columns={'index': 'Index', 0: 'Type'})
            data_types = data_types[data_types['Index'] != 'Target']  ##Removing the target from factorVariables list
            for Index, row in data_types.iterrows():
                if row['Type'] in ['object', 'str']:
                    factorVariables.append(row['Index'])

        if numericVariables is None or len(numericVariables) == 0:
            numericVariables = []
            data_types = Data.dtypes.reset_index().rename(columns={'index': 'Index', 0: 'Type'})
            data_types = data_types[data_types['Index'] != 'Target']  ##Removing the target from factorVariables list
            for Index, row in data_types.iterrows():
                if row['Type'] in ['int64', 'float64']:
                    numericVariables.append(row['Index'])

        ### Creating YMC Dictionaries for Factors (Creating the dictionaries) -----
        YMCFactorDictionaryList = dict()
        for variableToConvert in factorVariables:
            #print(variableToConvert)

            ## If There is only one factor in the variable we continiue to the next variable
            if (len(Data[variableToConvert].unique()) < 2):
                continue

            ## YMC for the factor
            meanDictionary = FunFactorYMC(VariableToConvert = variableToConvert, TargetName = 'Target',Data = Data, FrequencyNumber = 100, Fun = np.median, Suffix='_MeanFactorYMC' )
            medianDictionary = FunFactorYMC(VariableToConvert = variableToConvert, TargetName = 'Target',Data = Data, FrequencyNumber = 100, Fun = np.median, Suffix='_MedianFactorYMC' )
            Dictionary = meanDictionary.merge(medianDictionary, on = "Variable")

            # Inserting the dictionary into a list
            YMCFactorDictionaryList[variableToConvert] = Dictionary

            ### Delete all temporary Variables ----------------------------------------
            del variableToConvert
            del meanDictionary
            del medianDictionary
            del Dictionary

        ### Inserting the Total YMC Measures for all the new predictions ----------
        totalYMeanTarget = np.mean(Data['Target'])
        totalYMedianTarget = np.median(Data['Target'])

        ### Inserting the YMC Values from the dictionaries to the DataPanel -------
        for variableName in YMCFactorDictionaryList:
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

        ### Numerical Data Manipulation (YMC) -------------------------------------
        YMCDictionaryNumericList = dict()
        for variableToConvert in numericVariables:
            Variable = Data[variableToConvert].astype(float)
            Variable = Variable.fillna(0)

            ## If There is only one factor in the variable we continiue to the next variable
            if (len(Variable.unique()) < 2):
                continue
            
            ## YMC dictionary for numeric data
            YMCDictionaryNumericList[variableToConvert] = FunNumericYMC(Variable = Variable,Target = Data['Target'],NumberOfGroups = max(10, round(len(Variable) / 600)),Fun = np.mean,Name = "MeanNumericYMC")
        
            ### Delete all temporary Variables ----------------------------------------
            del Variable
            del variableToConvert

        ### Creating the YMC calculation for each numeric variable ----------------
        numericYMC = pd.DataFrame(data={})
        for variableToConvert in YMCDictionaryNumericList:
            Variable = pd.DataFrame(data={variableToConvert: Data[variableToConvert].astype(float)})
            Variable.loc[:, variableToConvert] = Variable[variableToConvert].fillna(0)

            # Inserting the numeric dictionary into VariableDictionary
            VariableDictionary = YMCDictionaryNumericList[variableToConvert]

            # Adding All the YMC
            VariableDictionary.index = pd.IntervalIndex.from_arrays(VariableDictionary['lag_value'],
                                                                    VariableDictionary['value'],
                                                                    closed='left')
            V = Variable[variableToConvert]
            Variable[['MeanNumericYMC']] = VariableDictionary.loc[V][['MeanNumericYMC']].reset_index(drop=True)

            # Creating YMC table
            YMC = pd.DataFrame(data={'variableToConvert_MeanNumericYMC': Variable['MeanNumericYMC']})

            # Left join YMC table to NUmeric_YMC table
            numericYMC = pd.concat([numericYMC, YMC], axis=1)
            numericYMC.columns = list(map(lambda x: x.replace('variableToConvert', variableToConvert, 1), numericYMC.columns))

            ### Delete all temporary Variables ----------------------------------------
            del variableToConvert
            del Variable
            del VariableDictionary
            del V
            del YMC

        ### Left join of Numeric_YMC table to the DataPanel -----------------------
        # Data = Data.join(Numeric_YMC.set_index('ClaimNo_Descision'), how='left', on='ClaimNo_Descision')
        Data = pd.concat([Data, numericYMC], axis=1)

        ### Delete all temporary Variables ----------------------------------------
        del numericYMC

        ### -----------------------------------------------------------------------
        ### ----------------------- Target Model ----------------------------------
        ### -----------------------------------------------------------------------
        ### Taking the YMC_Suspicious variables -----------------------------------
        YMCVariables = Data.columns[["_MeanNumericYMC" in i or "_MeanFactorYMC" in i or "_MedianFactorYMC" in i for i in Data.columns]]
    
        ### Creating Train Data for model -------------------------------------
        XTrain, XValidate, YTrain, YValidate = train_test_split(Data.loc[:, YMCVariables].astype(float), 
                                                                Data['Target'], 
                                                                test_size=float(conf['ValidationDataPercent']), 
                                                                random_state=0)

        ### Removing null from the model ---------------------
        XTrain = XTrain.loc[~np.isnan(YTrain), :].reset_index(drop=True)
        YTrain = YTrain.loc[~np.isnan(YTrain)].reset_index(drop=True)
        XValidate = XValidate.loc[~np.isnan(YValidate)].reset_index(drop=True)
        YValidate = YValidate.loc[~np.isnan(YValidate)].reset_index(drop=True)

        ### Defining the model ------------------------------------------------
        LGBEstimator = LightGBM.LGBMClassifier(boosting_type='gbdt',
                                               objective='binary')

        ### Defining the Grid -------------------------------------------------
        parameters = dict(num_leaves = stats.randint(10,500),
                          #panalty = ('l1','l2'),
                          #C = stats.uniform(0, 1),
                          n_estimators = stats.randint(10,400),
                          min_child_samples = stats.randint(0,20),
                          max_depth = stats.randint(1,15),
                          learning_rate = stats.uniform(0,1),
                          reg_alpha = stats.uniform(0,1))

        ## K-Fold Cross-Valisdation -------------------------------------------
        KF = KFold(n_splits = 5, shuffle = True, random_state=4)

        ### Run the model -----------------------------------------------------
        GBMGridSearch = RandomizedSearchCV(estimator = LGBEstimator,
                                            param_distributions = parameters,
                                            scoring='accuracy',#'accuracy',,‘neg_mean_absolute_error’,'neg_root_mean_squared_error
                                            n_iter = 120,##Number of Triels to find the best grid
                                            n_jobs = 4,
                                            cv = KF, #k-fold number
                                            refit = True,
                                            random_state = 4
                                            )
        GBMModel = GBMGridSearch.fit(X=XTrain, y=YTrain)

        ### Fitting the best model --------------------------------------------
        GBMModel = GBMModel.best_estimator_.fit(X=XTrain, y=YTrain)

        ### Saving the features name ------------------------------------------
        GBMModel.feature_names = list(YMCVariables)
        GBMModel.NameColumnsOfDataInModel =  list(map(lambda x: x.replace('_MeanNumericYMC', '').replace('_MeanFactorYMC', '').replace('_MedianFactorYMC', ''), GBMModel.feature_names))
        NameColumnsOfDataInModel = GBMModel.NameColumnsOfDataInModel 

        ## Maximum and Minimum Y ----------------------------------------------
        maxY = np.max(YTrain)
        minY = np.min(YTrain)

        ### -------------------------------------------------------------------
        ### ------------ Metrics for estimating the model ---------------------
        ###--------------------------------------------------------------------
        ## Validate Predictions -----------------------------------------------
        XValidate = XValidate.loc[:,GBMModel.feature_names].astype(float)
        YValidateProba = GBMModel.predict_proba(XValidate)[:,1]
        YValidateProba = YValidateProba.astype(float)
        YValidateProba = np.where(YValidateProba >= maxY, maxY, YValidateProba)
        YValidateProba = np.where(YValidateProba <= minY, minY, YValidateProba)
        YValidateClass = np.where(YValidateProba >= YValidateProba.mean(),1,0)

        modelMetricsEstimation = {}
        modelMetricsEstimation['Accuracy'] = metrics.accuracy_score(YValidate, YValidateClass)
        modelMetricsEstimation['F1'] = metrics.f1_score(YValidate, YValidateClass)
        modelMetricsEstimation['Precision'] = metrics.precision_score(YValidate, YValidateClass)
        modelMetricsEstimation['Recall'] = metrics.recall_score(YValidate, YValidateClass)
        modelMetricsEstimation['PrecisionRecallCurve']  = metrics.precision_recall_curve(YValidate, YValidateProba)
        modelMetricsEstimation['RocCurve']  = metrics.roc_curve(YValidate, YValidateProba)

        ## Logistic Regression ------------------------------------------------
        logisticRegressionModel = LogisticRegression(max_iter=1000).fit(XTrain.loc[:,GBMModel.feature_names].astype(float), YTrain)
        XTrain['PredictLogisticRegression'] = logisticRegressionModel.predict_proba(XTrain.loc[:,GBMModel.feature_names].astype(float))[:,1]
        
        ## Train Predictions --------------------------------------------------
        XTrain['PredictGBM'] = GBMModel.predict_proba(XTrain.loc[:,GBMModel.feature_names].astype(float))[:,1]
        XTrain['PredictGBM'] = XTrain['PredictGBM'].astype(float)
        XTrain['PredictGBM'] = np.where(XTrain['PredictGBM']>=maxY,maxY,XTrain['PredictGBM'])
        XTrain['PredictGBM'] = np.where(XTrain['PredictGBM']<=minY,minY,XTrain['PredictGBM'])
   
        ## Ranks ---------------------------------------------------------------
        predictionsDictionary = Ranks_Dictionary(RJitter(XTrain['PredictGBM'],0.00001), ranks_num = 10)
        predictionsDictionary.index = pd.IntervalIndex.from_arrays(predictionsDictionary['lag_value'],
                                                                  predictionsDictionary['value'],
                                                                  closed='left')
        # Convert Each prediction value to rank
        XTrain['Rank'] = predictionsDictionary.loc[XTrain['PredictGBM']]['rank'].reset_index(drop=True)

        ### Current Time -------------------------------------------------------
        now = datetime.now() 
        CreateModelDate = now.strftime("%Y-%m-%d %H:%M:%S")

        ### Save The Model ----------------------------------------------------- 
        Path = conf['Path']
        #os.getcwd()
        #Path = Path.replace('Segment', Segment, 1)  
        f = open(Path, 'wb')
        pickle.dump([factorVariables,
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
                    conf,
                    modelMetricsEstimation], f)

        f.close()

        ### Output ----------------------------------------------------------------
        Output = pd.DataFrame(data={'Target': YTrain,
                                    'PredictGBM': XTrain['PredictGBM'],
                                    'PredictLogisticRegression': XTrain['PredictLogisticRegression'],
                                    'Rank': XTrain['Rank']})

        return Output

#Check The Model --------------------------------------------------------  
# from sklearn.datasets import make_classification
# makeClassificationX,makeClassificationY = make_classification(n_samples = 5000,class_sep = 4,random_state=0)
# Data = pd.DataFrame(makeClassificationX)
# Data['Y'] = pd.DataFrame(makeClassificationY)

# conf={
#     'Path':'/Users/dhhazanov/UmAI/Models/Model.pckl',
#     'Target':'Y',
#     'ColumnSelectionType': 'Drop',#Drop,Keep
#     'Keep': None,#['GENDER', 'FAMILY_STATUS','GIL'],
#     'Drop': ['GIL','ISUK_MERAKEZ','FAMILY_STATUS','ISHUN','M_CHOD_TASHLOM_BR'],#None,
#     'ModelType': None #GBM,Linear regression,...
# }
# RunModel = Model(Data,conf,logger)
# Output = RunModel.fit()

# Output.groupby('Target')['PredictGBM'].apply(np.mean).reset_index()
# Output.groupby('Rank')['PredictGBM'].apply(np.mean).reset_index()
# Output.groupby('Rank')['Target'].apply(np.mean).reset_index()
# Output.groupby('Rank')['PredictLogisticRegression'].apply(np.mean).reset_index()

#Check The Model 2 --------------------------------------------------------  
# Data = pd.read_csv('/Users/dhhazanov/UmAI/Eli_data_health.csv')
# Data.head()
# conf={
#     'Path':'/Users/dhhazanov/UmAI/Models/Model.pckl',
#     'Target':'Y',
#     'ColumnSelectionType': 'Drop',#Drop,Keep
#     'Keep': None,#['GENDER', 'FAMILY_STATUS','GIL'],
#     'Drop': ['GIL','Unnamed: 0'],#None,
#     'ModelType': None #GBM,Linear regression,...
# }
# Data['Y'] = np.where(Data['GIL'] >= Data['GIL'].mean(),1,0)

# RunModel = Model(Data,conf,logger)
# Output = RunModel.fit()         


# Output.groupby('Target')['PredictGBM'].apply(np.mean).reset_index()
# Output.groupby('Rank')['PredictGBM'].apply(np.mean).reset_index()
# Output.groupby('Rank')['Target'].apply(np.mean).reset_index()
# Output.groupby('Rank')['PredictLogisticRegression'].apply(np.mean).reset_index() 


## Read Data from local memory ---------------------------------------------------------------------------------
# Data = pd.read_csv('/Users/dhhazanov/UmAI/ppp.parquet_1_for_model.csv')
# Data = Data.head(1000)
# Data['NewHistoricalYTarget'] = np.where(Data['GIL'] >= Data['GIL'].mean(),1,0)
# conf={
#     'Path':'/Users/dhhazanov/UmAI/Models/Model.pckl',
#     'Target':'NewHistoricalYTarget',
#     'ColumnSelectionType': 'Drop',#Drop,Keep
#     'Keep': None,#['GENDER', 'FAMILY_STATUS','GIL'],
#     'Drop': ['priditScore'], 
#     'ModelType': None #GBM,Linear regression,...
# }
# # conf = {'Target': 'NewHistoricalYTarget',
# #  'ColumnSelectionType': 'Drop', 
# #  'Keep': None, 
# #  'Drop': ['priditScore'], 
# #  'ModelType': 'GBM', 
# #  'DataFileType': 'parquet', 
# #  'DataPath': 'C:/git/Vehicle_suspicious_score_prod/uploads//Umai/Admin-/0/ppp.parquet_1_for_model.gzip', 
# #  'ScorePath': 'C:/git/Vehicle_suspicious_score_prod/uploads//Umai/Admin-/0/ppp.parquet_1_PriditScore'}
# import re
# Data = Data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))  
# RunModel = Model(Data,conf,logger)
# Output = RunModel.fit()    
# # Data = pd.read_parquet(r'C:\github\Utilities\machine_learning_examples\ppp_v1.parquet.gzip', engine='pyarrow')
