
import os
import pickle
from datetime import datetime

import lightgbm as LightGBM
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import QuantileRegressor
from Functions import Ranks_Dictionary, RJitter, FunFactorYMC, FunNumericYMC
pd.options.display.float_format = '{:.2f}'.format
import warnings
import logging as logger
import statsmodels.formula.api as smf

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
class NumericModel():
    
    ## Ranks Dictionary -----------------------------------------------------------
    def __init__(self, Data, conf, logger):
        self.Data = Data
        self.conf = conf
        self.logger = logger

        self.logger.debug('Model was created')

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
        if (conf['ColumnSelectionType'] != None and conf['Drop'] == None):
            logger.info('ColumnSelectionType not found in conf using default for \"Drop\" -> []')
            conf['Drop'] = []

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
        ### ----------------------- Light GBM Model -------------------------------
        ### -----------------------------------------------------------------------
        ### Taking the variables -----------------------------------
        YMCVariables = Data.columns[["_MeanNumericYMC" in i or "_MeanFactorYMC" in i or "_MedianFactorYMC" in i for i in Data.columns]]
        #YMCVariables = (*YMCVariables, *numericVariables)
    
        ### Creating Train Data for model -------------------------------------
        XTrain = Data.loc[:, YMCVariables].astype(float)
        YTrain = Data['Target']

        ### Removing null from the model ---------------------
        XTrain = XTrain.loc[~np.isnan(YTrain), :].reset_index(drop=True)
        YTrain = YTrain.loc[~np.isnan(YTrain)].reset_index(drop=True)

        ### Defining the model ------------------------------------------------
        # objective = 'multiclass'
        LGBEstimator = LightGBM.LGBMRegressor(boosting_type='gbdt',
                                              objective='regression')
        # LGBEstimator = LightGBM.LGBMClassifier(boosting_type='gbdt',
        #                                        objective='binary')

        ### Defining the Grid -------------------------------------------------
        # parameters = {'num_leaves':[20,40,60,80,100], 
        #               #'panalty': ['l1','l2'],
        #               #'C': [0, 0.3, 0.5, 1],
        #               'n_estimators': [50,100,150,200,300],
        #               'min_child_samples':[5,10,15],
        #               'max_depth':[-1,5,10,20,30,40,45],
        #               'learning_rate':[0.05,0.1,0.2],
        #               'reg_alpha':[0,0.01,0.03]}
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
                                            scoring='neg_root_mean_squared_error',#'accuracy',,‘neg_mean_absolute_error’,'neg_root_mean_squared_error
                                            n_iter = 120,##Number of Triels to find the best grid
                                            n_jobs = 4,
                                            cv = KF, #k-fold number
                                            refit = True,
                                            random_state = 4
                                            )
        GBMModel = GBMGridSearch.fit(X=XTrain, y=YTrain)
  
        ### Fitting the best model --------------------------------------------
        GBMModel = GBMModel.best_estimator_.fit(X=XTrain, y=YTrain)
        # GBMModel.best_params_ #{'learning_rate': 0.06838596828617904, 'max_depth': 14, 'min_child_samples': 0, 'n_estimators': 195, 'num_leaves': 310, 'reg_alpha': 0.30033842756085605}
        # results = pd.DataFrame(GBMModel.cv_results_)[['params','mean_test_score','std_test_score']]
        # results.sort_values(by="mean_test_score",ascending=False,inplace=True)
        # results.reset_index(drop=False,inplace=True)
        # results['mean_test_score'].plot(yerr=[results['std_test_score'], results['std_test_score']],subplots=True)
        
        ### Saving the features name ------------------------------------------
        GBMModel.feature_names = list(YMCVariables)
        GBMModel.NameColumnsOfDataInModel =  list(map(lambda x: x.replace('_MeanNumericYMC', '').replace('_MeanFactorYMC', '').replace('_MedianFactorYMC', ''), GBMModel.feature_names))
        NameColumnsOfDataInModel = GBMModel.NameColumnsOfDataInModel 

        ## Maximum and Minimum Y ----------------------------------------------
        maxY = np.max(YTrain)
        minY = np.min(YTrain)

        ## Delete variables ---------------------------------------------------
        del YMCVariables
        del parameters
        del LGBEstimator
        del XTrain
        del YTrain
        del GBMGridSearch
        del KF
        
        ## Predictions ----------------------------------------------------
        XTrain = Data.loc[:,GBMModel.feature_names].astype(float)
        Data['PredictGBM'] = GBMModel.predict(XTrain)##for GBM regressor
        ##Data['PredictGBM'] = GBMModel.predict_proba(XTrain)[:,1]
        Data['PredictGBM'] = Data['PredictGBM'].astype(float)
        Data['PredictGBM'] = np.where(Data['PredictGBM']>=maxY,maxY,Data['PredictGBM'])
        Data['PredictGBM'] = np.where(Data['PredictGBM']<=minY,minY,Data['PredictGBM'])
   
        ## Ranks ----------------------------------------------------
        predictionsDictionary = Ranks_Dictionary(RJitter(Data['PredictGBM'],0.00001), ranks_num = 10)
        predictionsDictionary.index = pd.IntervalIndex.from_arrays(predictionsDictionary['lag_value'],
                                                                  predictionsDictionary['value'],
                                                                  closed='left')
        # Convert Each prediction value to rank
        Data['Rank'] = predictionsDictionary.loc[Data['PredictGBM']]['rank'].reset_index(drop=True)


        ### -----------------------------------------------------------------------
        ### ----------------------- Calibration Model -----------------------------
        ### -----------------------------------------------------------------------
        ### Taking the variables -----------------------------------
        YMCVariables = Data.columns[["_MeanNumericYMC" in i or "_MeanFactorYMC" in i or "_MedianFactorYMC" in i for i in Data.columns]]

        ### Creating Train Data for model -------------------------------------
        XTrain = Data.loc[:, YMCVariables].astype(float)
        YTrain = Data['Target']

        ### Removing null from the model ---------------------
        XTrain = XTrain.loc[~np.isnan(YTrain), :].reset_index(drop=True)
        YTrain = YTrain.loc[~np.isnan(YTrain)].reset_index(drop=True)

        ### Defining the model ------------------------------------------------
        LGBEstimator = LightGBM.LGBMRegressor(boosting_type='gbdt',
                                              alpha=0.5,
                                              metric = 'quantile',
                                              objective='quantile')

        ### Defining the Grid -------------------------------------------------
        parameters = dict(num_leaves = stats.randint(10,500),
                          #panalty = ('l1','l2'),
                          #C = stats.uniform(0, 1),
                          n_estimators = stats.randint(10,400),
                          min_child_samples = stats.randint(0,20),
                          max_depth = stats.randint(1,7),
                          learning_rate = stats.uniform(0,1),
                          reg_alpha = stats.uniform(0,1))

        ## K-Fold Cross-Valisdation -------------------------------------------
        KF = KFold(n_splits = 5, shuffle = True, random_state=4)

        ### Run the model -----------------------------------------------------
        GBMGridSearch = RandomizedSearchCV(estimator = LGBEstimator,
                                            param_distributions = parameters,
                                            scoring= 'neg_mean_absolute_error',#'accuracy',,‘neg_mean_absolute_error’,'neg_root_mean_squared_error
                                            n_iter = 120,##Number of Triels to find the best grid
                                            n_jobs = 4,
                                            cv = KF, #k-fold number
                                            refit = True,
                                            random_state = 4
                                            )
        QrModel = GBMGridSearch.fit(X = XTrain, y = YTrain)
  
        ### Fitting the best model --------------------------------------------
        QrModel = QrModel.best_estimator_.fit(X = XTrain, y = YTrain)

        ## Prediction for calibration------------------------------------------
        Data['PredictQuantile'] = QrModel.predict(XTrain)
        #Data['PredictQuantile'] = np.maximum(1, Data['PredictQuantile'])
        
        ##LTV Upper Cut Point
        UpperBorder = np.quantile(Data['PredictQuantile'],0.999)
        UpperValue = np.mean(Data.loc[Data['PredictQuantile'] >= UpperBorder,:]['PredictQuantile'])
        Data['PredictQuantile'] = np.where(Data['PredictQuantile'] >= UpperBorder, UpperValue, Data['PredictQuantile'])
                
        # Calibration ----------------------------------------------------------
        pred = Data['PredictQuantile'][Data['PredictQuantile']<np.quantile(Data['PredictQuantile'],0.9)]
        predTop = Data['PredictQuantile'][Data['PredictQuantile']>=np.quantile(Data['PredictQuantile'],0.9)]
        
        if (len(pred)==0 or len(predTop)==0):
            Data['PredictQuantile'] = Data['PredictQuantile']+np.random.uniform(0, 0.001, len(Data['PredictQuantile']))
            Data['PredictQuantile'] = Data['PredictQuantile']+np.random.uniform(0, 0.00001, len(Data['PredictQuantile']))
            Data['PredictQuantile'] = Data['PredictQuantile']+np.random.uniform(0, 0.000001, len(Data['PredictQuantile']))
            Data['PredictQuantile'] = Data['PredictQuantile']+np.random.uniform(0, 0.0000001, len(Data['PredictQuantile']))
            Data['PredictQuantile'] = Data['PredictQuantile']+np.random.uniform(0, 0.0000001, len(Data['PredictQuantile']))
            Data['PredictQuantile'] = Data['PredictQuantile']+np.random.uniform(0, 0.0000001, len(Data['PredictQuantile']))

            pred = Data['PredictQuantile'][Data['PredictQuantile']<np.quantile(Data['PredictQuantile'],0.9)]
            predTop = Data['PredictQuantile'][Data['PredictQuantile']>=np.quantile(Data['PredictQuantile'],0.9)]        
    
        Calibration1 = Ranks_Dictionary(pred,max(10,round(len(pred)/700)))
        Calibration2 = Ranks_Dictionary(predTop,max(5,round(len(predTop)/400)))
        
        Calibration1.iloc[len(Calibration1)-1,1] = Calibration2.loc[Calibration2['rank']==1,'value'][0]
        Calibration2 = Calibration2.loc[Calibration2['rank']>1,:]   
        
        Calibration = pd.concat([Calibration1, Calibration2]) 
        Calibration['rank'] = np.linspace(start=1,stop=len(Calibration),num=len(Calibration),dtype=int)
        
        # Creating interval index for fast location 
        Calibration.index = pd.IntervalIndex.from_arrays(Calibration['lag_value'],
                                                        Calibration['value'],
                                                        closed='left')
        
        # Convert Each value in variable to rank
        Data['PredictedRank'] = Calibration.loc[Data['PredictQuantile']]['rank'].reset_index(drop=True)
        
        
        a = Data.groupby('PredictedRank').agg(mean = ('PredictQuantile', 'mean'),length = ('PredictQuantile', 'count')).reset_index()
        b = Data.groupby('PredictedRank').agg(mean = ('Target', 'mean'),length = ('PredictQuantile', 'count')).reset_index()
        a.columns = ["Ranks","PredictedMean","PredictedLength"]
        b.columns = ["Ranks","YMean","YLength"]

        c = pd.merge(a, b, on='Ranks', how='left')
        c['Diff'] = np.abs(100*(c['PredictedMean']-c['YMean'])/c['YMean'])
        
        CalibrationTable = pd.DataFrame(dict(rank=c['Ranks'],
                                            Calibration=c['YMean']/c['PredictedMean'],
                                            YMean=c['YMean'],
                                            length=c['YLength']))
        Calibration = pd.merge(Calibration,CalibrationTable,on='rank', how='left')
        Calibration.loc[Calibration['Calibration'] == float("inf"),'Calibration'] = 1
        Calibration.loc[np.isnan(Calibration['Calibration']),'Calibration'] = np.median(Calibration['Calibration'].dropna())

        del a,b,c,CalibrationTable,pred,predTop

        ###Second calibration (Smoothing the calibration)
        Calibration['Calibration2'] = Calibration['Calibration'].rolling(window=3, min_periods=1).mean()
        Calibration.index = pd.IntervalIndex.from_arrays(Calibration['lag_value'],
                                                Calibration['value'],
                                                closed='left')
        Data['Calibration'] = Calibration.loc[Data['PredictQuantile']]['Calibration'].reset_index(drop=True)
        Data['PredictCalibrated'] = Data['Calibration'] * Data['PredictQuantile']


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
                    GBMModel, maxY, minY,
                    predictionsDictionary,
                    QrModel, UpperBorder, UpperValue, Calibration,
                    CreateModelDate,
                    NameColumnsOfDataInModel,
                    conf], f)

        f.close()

        ### Output ----------------------------------------------------------------
        Output = pd.DataFrame(data={'Target': Data['Target'],
                                    'PredictGBM': Data['PredictGBM'],
                                    'PredictQuantile': Data['PredictQuantile'],
                                    'PredictCalibrated': Data['PredictCalibrated'],
                                    'Rank': Data['Rank']})

        return Output


## Read Data from local memory ---------------------------------------------------------------------------------
# Data = pd.read_csv('/Users/dhhazanov/UmAI/Data/insurance_claims.csv')
# conf={
#     'Path':'/Users/dhhazanov/UmAI/Models/Model.pckl',
#     'Target':'total_claim_amount',
#     'ColumnSelectionType': 'Drop',#Drop,Keep
#     'Keep': None,#['GENDER', 'FAMILY_STATUS','GIL'],
#     'Drop': ['priditScore'], 
#     'ModelType': None #GBM,Linear regression,...
# }

# import re
# Data = Data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))  
# RunModel = NumericModel(Data,conf,logger)
# Output = RunModel.fit()    
