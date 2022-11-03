
import os
import pickle
from datetime import datetime
from docutils import DataError

import lightgbm as LightGBM
import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV, KFold

from Functions import Ranks_Dictionary, RJitter, FunFactorYMC, FunNumericYMC
pd.options.display.float_format = '{:.2f}'.format
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
    ## Ranks Dictionary ----------------------------------------------------------
    def __init__(self, Data, conf):
        self.Data = Data
        self.conf = conf

    def fit(self):
        Data = self.Data
        conf = self.conf

        ## Fill Configuration ------------------------------------------------------
        if (not 'factorVariables' in conf or conf['factorVariables'] == None):
            conf['factorVariables'] = []
            factorVariables = conf['factorVariables']
        if (not 'numericVariables' in conf or conf['numericVariables'] == None):
            conf['numericVariables'] = []
            numericVariables = conf['numericVariables']
        if (not 'ColumnSelectionType' in conf or conf['ColumnSelectionType'] == None):
            conf['ColumnSelectionType'] = []
            conf['Keep'] = []
            conf['Drop'] = []
        if (conf['ColumnSelectionType'] != None and conf['Keep']==None):
            conf['Keep'] = []
        if (conf['ColumnSelectionType'] != None and conf['Drop']==None):
            conf['Drop'] = []

        ## Save Target columns -----------------------------------------------------
        Target = Data[conf['Target']]
        try:
            Data = Data.drop(conf['Target'],axis=1)
        except Exception:           
            print("Target column is not exist")   

        ## Drop ot Select columns from Data ----------------------------------------
        if(len(conf['ColumnSelectionType'])!=0 and len(conf['Keep'])!=0):     
            try:
                Data = Data.loc[:,np.intersect1d(Data.columns.values,conf['Keep'])]
                print('Selected chosen columns')
            except Exception:           
                print("Didn't selected anything")         
                raise  
        if(len(conf['ColumnSelectionType'])!=0 and len(conf['Drop'])!=0):
            try:
                Data = Data.drop(conf['Drop'],axis=1)
                print('Droped selected columns')
            except Exception:           
                print("Didn't drop any columns")         
                raise  

        ## Insert Target columns -----------------------------------------------------
        Data['Target'] = Target
        del Target

        ## factorVariables and numericVariables Variables --------------------------
        factorVariables = conf['factorVariables']
        numericVariables = conf['numericVariables']

        ## Fill the FactorVariables and NumericVariables list ----------------------
        if factorVariables is None or len(factorVariables)==0:
            factorVariables = []
            data_types = Data.dtypes.reset_index().rename(columns={'index': 'Index', 0: 'Type'})
            data_types = data_types[data_types['Index']!='Target']##Removing the target from factorVariables list
            for Index, row in data_types.iterrows():
                if row['Type'] in ['object', 'str']:
                    factorVariables.append(row['Index'])

        if numericVariables is None or len(numericVariables)==0:
            numericVariables = []
            data_types = Data.dtypes.reset_index().rename(columns={'index': 'Index', 0: 'Type'})
            data_types = data_types[data_types['Index']!='Target']##Removing the target from factorVariables list
            for Index, row in data_types.iterrows():
                if row['Type'] in ['int64', 'float64']:
                    numericVariables.append(row['Index'])

                
        ### Creating YMC Dictionaries for Factors (Creating the dictionaries) -----
        YMCFactorDictionaryList = dict()
        for variableToConvert in factorVariables:
            #print(variableToConvert)
            ##Creating the YMC Dictionaries
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
                                    variableName+"_MeanFactorYMC",
                                    variableName+"_MedianFactorYMC"]
        
            Data = Data.join(YMCDictionary.set_index([variableName]), how='left', on=[variableName])
            Data.loc[:, variableName+"_MeanFactorYMC"] = Data[variableName+"_MeanFactorYMC"].fillna(totalYMeanTarget)        
            Data.loc[:, variableName+"_MedianFactorYMC"] = Data[variableName+"_MedianFactorYMC"].fillna(totalYMedianTarget)
        
            ### Delete all temporary Variables ----------------------------------------       
            del YMCDictionary
            del variableName 


        ### Numerical Data Manipulation (YMC) -------------------------------------
        YMCDictionaryNumericList = dict()
        for variableToConvert in numericVariables:
            Variable = Data[variableToConvert].astype(float)
            Variable = Variable.fillna(0)

            YMCDictionaryNumericList[variableToConvert] = FunNumericYMC(Variable = Variable,Target = Data['Target'],NumberOfGroups = 10,Fun = np.mean,Name = "MeanNumericYMC")
        
            ### Delete all temporary Variables ----------------------------------------
            del Variable
            del variableToConvert

        ### Creating the YMC calculation for each numeric variable ----------------
        numericYMC = pd.DataFrame(data={})
        for variableToConvert in numericVariables:
            Variable = pd.DataFrame(data={variableToConvert: Data[variableToConvert].astype(float)})
            Variable.loc[:,variableToConvert] = Variable[variableToConvert].fillna(0)
        
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
        #Data = Data.join(Numeric_YMC.set_index('ClaimNo_Descision'), how='left', on='ClaimNo_Descision')
        Data = pd.concat([Data,numericYMC], axis = 1)
        
        ### Delete all temporary Variables ----------------------------------------
        del numericYMC 

        
        ### -----------------------------------------------------------------------
        ### ----------------------- Target Model ----------------------------------
        ### -----------------------------------------------------------------------
        ### Taking the YMC_Suspicious variables -----------------------------------

        YMCVariables = Data.columns[["_MeanNumericYMC" in i or "_MeanFactorYMC" in i or "_MedianFactorYMC" in i for i in Data.columns]]
        YMCVariables = (*YMCVariables,*numericVariables)
        
        ### Creating Train Data for model -------------------------------------
        XTrain = Data.loc[:,YMCVariables].astype(float)
        YTrain = Data['Target']
        
        ### Removing null from the model ---------------------
        XTrain = XTrain.loc[~np.isnan(YTrain),:].reset_index(drop=True)
        YTrain = YTrain.loc[~np.isnan(YTrain)].reset_index(drop=True)

        ### Defining the model ------------------------------------------------
        LGBEstimator = LightGBM.LGBMRegressor(boosting_type = 'gbdt',
                                              objective = 'regression')
        
        ### Defining the Grid -------------------------------------------------
        parameters = {'num_leaves':[20,40,60,80,100], 
                      #'panalty': ['l1','l2'],
                      #'C': [0, 0.3, 0.5, 1],
                      'n_estimators': [50,100,150,200,300],
                      'min_child_samples':[5,10,15],
                      'max_depth':[-1,5,10,20,30,40,45],
                      'learning_rate':[0.05,0.1,0.2],
                      'reg_alpha':[0,0.01,0.03]}

        ## K-Fold Cross-Valisdation -------------------------------------------
        KF = KFold(n_splits = 5, shuffle = True, random_state=4)

        ### Run the model -----------------------------------------------------
        GBMGridSearch = RandomizedSearchCV(estimator = LGBEstimator,
                                                param_distributions = parameters,
                                                scoring='neg_mean_absolute_error',#'accuracy',,‘neg_mean_absolute_error’,'neg_root_mean_squared_error
                                                n_iter=30,
                                                n_jobs = 4,
                                                cv = KF, #k-fold number
                                                refit = True
                                                )
        GBMModel = GBMGridSearch.fit(X=XTrain, y=YTrain)   
        
        ### Fitting the best model --------------------------------------------
        GBMModel = GBMModel.best_estimator_.fit(X=XTrain, y=YTrain)
        # GBMModel.best_params_
        # results = pd.DataFrame(GBMModel.cv_results_)[['params','mean_test_score','std_test_score']]
        # results.sort_values(by="mean_test_score",ascending=False,inplace=True)
        # results.reset_index(drop=False,inplace=True)
        # results['mean_test_score'].plot(yerr=[results['std_test_score'], results['std_test_score']],subplots=True)
        ### Saving the features name ------------------------------------------
        GBMModel.feature_names = list(YMCVariables)
        
        del YMCVariables
        del parameters
        del LGBEstimator
        del XTrain
        del YTrain
        del GBMGridSearch
        del KF

        ### Current Time ----------------------------------------------------------
        now = datetime.now() 
        CreateModelDate = now.strftime("%Y-%m-%d %H:%M:%S")

        ### Save The Model --------------------------------------------------------  
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
                    CreateModelDate], f)

        f.close()

        ### -----------------------------------------------------------------------
        ### ----------------------- Prediction Model ------------------------------
        ### -----------------------------------------------------------------------
        ### Taking the YMC_Suspicious variables -----------------------------------

        XTest = Data.loc[:,GBMModel.feature_names].astype(float)
        Data['predictGBM'] = GBMModel.predict(XTest)
        Data['predictGBM'] = Data['predictGBM'].astype(float)

        ### Output ----------------------------------------------------------------
        Output = pd.DataFrame(data={'Target': Data['Target'],
                                    'predictGBM': Data['predictGBM']})

        return Output


# ## Check The Model --------------------------------------------------------  
# Data = pd.read_parquet('/Users/dhhazanov/Downloads/ppp_v1.parquet.gzip', engine='pyarrow')
# #Data['Target'] = np.where(Data['GIL'] >= Data['GIL'].mean(),1,0)
# Data['Target2'] = np.where(Data['GIL'] >= Data['GIL'].mean(),1,0)
# conf={
#     'Path':'/Users/dhhazanov/UmAI/Models/Model.pckl',
#     'Target':'Target2',
#     'ColumnSelectionType': 'Drop',#Drop,Keep
#     'Keep': None,#['GENDER', 'FAMILY_STATUS','GIL'],
#     'Drop': ['GIL','ISUK_MERAKEZ','FAMILY_STATUS','ISHUN','M_CHOD_TASHLOM_BR'],#None,
#     'ModelType': None #GBM,Linear regression,...
# }
# RunModel = Model(Data,conf)
# Output = RunModel.fit()

# Output.groupby('Target')['predictGBM'].apply(np.mean).reset_index()
