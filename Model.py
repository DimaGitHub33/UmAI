
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

def Model(Data,conf={}):
    ## Fill Configuration -----------------------------------------------------
    if (not 'factorVariables' in conf or conf['factorVariables'] == None):
        conf['factorVariables'] = []
        factorVariables = conf['factorVariables']
    if (not 'numericVariables' in conf or conf['numericVariables'] == None):
        conf['numericVariables'] = []
        numericVariables = conf['numericVariables']

    ## factorVariables and numericVariables Variables --------------------------
    factorVariables = conf['factorVariables']
    numericVariables = conf['numericVariables']

    ## Fill the FactorVariables and NumericVariables list ----------------------
    if factorVariables is None or len(factorVariables)==0:
        factorVariables = []
        data_types = Data.dtypes.reset_index().rename(columns={'index': 'Index', 0: 'Type'})
        for Index, row in data_types.iterrows():
            if row['Type'] in ['object', 'str']:
                factorVariables.append(row['Index'])

    if numericVariables is None or len(numericVariables)==0:
        numericVariables = []
        data_types = Data.dtypes.reset_index().rename(columns={'index': 'Index', 0: 'Type'})
        for Index, row in data_types.iterrows():
            if row['Type'] in ['int64', 'float64']:
                numericVariables.append(row['Index'])

            
    ### Creating YMC Dictionaries for Factors (Creating the dictionaries) -----
    YMC_Factor_Dictionary_List = dict()
    for variableToConvert in factorVariables:
        #print(variableToConvert)
        ##Creating the YMC Dictionaries
        meanDictionary = FunFactorYMC(VariableToConvert = variableToConvert, TargetName = 'Target',Data = Data, FrequencyNumber = 100, Fun = np.median, Suffix='_MeanFactorYMC' )
        medianDictionary = FunFactorYMC(VariableToConvert = variableToConvert, TargetName = 'Target',Data = Data, FrequencyNumber = 100, Fun = np.median, Suffix='_MedianFactorYMC' )
        Dictionary = meanDictionary.merge(medianDictionary, on = "Variable")

        # Inserting the dictionary into a list
        YMC_Factor_Dictionary_List[variableToConvert] = Dictionary
        

    ### Delete all temporary Variables ----------------------------------------
    del variableToConvert
    del meanDictionary
    del medianDictionary
    del Dictionary
        
    ### Inserting the Total YMC Measures for all the new predictions ----------
    totalYMeanTarget = np.mean(Data['Target'])   
    totalYMedianTarget = np.median(Data['Target'])   

    ### Inserting the YMC Values from the dictionaries to the DataPanel -------
    for variableName in YMC_Factor_Dictionary_List:
        # variableName="M_ERUAS"
        Data.loc[:, variableName] = Data[variableName].astype(str)
        
        YMC_Dictionary = YMC_Factor_Dictionary_List[variableName]
        YMC_Dictionary.columns = [variableName, 
                                  variableName+"_MeanFactorYMC",
                                  variableName+"_MedianFactorYMC"]
    
        Data = Data.join(YMC_Dictionary.set_index([variableName]), how='left', on=[variableName])
        Data.loc[:, variableName+"_MeanFactorYMC"] = Data[variableName+"_MeanFactorYMC"].fillna(totalYMeanTarget)        
        Data.loc[:, variableName+"_MedianFactorYMC"] = Data[variableName+"_MedianFactorYMC"].fillna(totalYMedianTarget)
    
    ### Delete all temporary Variables ----------------------------------------
    del YMC_Dictionary
    del variableName 
        
    ### Numerical Data Manipulation (YMC) -------------------------------------
    YMC_Dictionary_Numeric_List = dict()
    for variableToConvert in numericVariables:
        Variable = Data[variableToConvert].astype(float)
        Variable = Variable.fillna(0)

        YMC_Dictionary_Numeric_List[variableToConvert] = FunNumericYMC(Variable = Variable,Target = Data['Target'],NumberOfGroups = 10,Fun = np.mean,Name = "MeanNumericYMC")
    
    ### Delete all temporary Variables ----------------------------------------
    del Variable
    del variableToConvert

    ### Creating the YMC calculation for each numeric variable ----------------
    Numeric_YMC = pd.DataFrame(data={})
    for variableToConvert in numericVariables:
        Variable = pd.DataFrame(data={variableToConvert: Data[variableToConvert].astype(float)})
        Variable.loc[:,variableToConvert] = Variable[variableToConvert].fillna(0)
    
        # Inserting the numeric dictionary into VariableDictionary
        VariableDictionary = YMC_Dictionary_Numeric_List[variableToConvert]
    
        # Adding All the YMC
        VariableDictionary.index = pd.IntervalIndex.from_arrays(VariableDictionary['lag_value'],
                                                                VariableDictionary['value'],
                                                                closed='left')
        V = Variable[variableToConvert]
        Variable[['MeanNumericYMC']] = VariableDictionary.loc[V][['MeanNumericYMC']].reset_index(drop=True)
    
        # Creating YMC table
        YMC = pd.DataFrame(data={'variableToConvert_MeanNumericYMC': Variable['MeanNumericYMC']})
        
        # Left join YMC table to NUmeric_YMC table
        Numeric_YMC = pd.concat([Numeric_YMC, YMC], axis=1)
        Numeric_YMC.columns = list(map(lambda x: x.replace('variableToConvert', variableToConvert, 1), Numeric_YMC.columns))
    
    ### Left join of Numeric_YMC table to the DataPanel -----------------------
    #Data = Data.join(Numeric_YMC.set_index('ClaimNo_Descision'), how='left', on='ClaimNo_Descision')
    Data = pd.concat([Data,Numeric_YMC], axis = 1)
    
    ### Delete all temporary Variables ----------------------------------------
    del variableToConvert
    del Numeric_YMC 
    del Variable
    del VariableDictionary
    del V
    del YMC
    
    
    ### -----------------------------------------------------------------------
    ### ----------------------- Target Model ----------------------------------
    ### -----------------------------------------------------------------------
    ### Taking the YMC_Suspicious variables -----------------------------------

    YMC_Variables = Data.columns[["_MeanNumericYMC" in i or "_MeanFactorYMC" in i or "_MedianFactorYMC" in i for i in Data.columns]]
    YMC_Variables = (*YMC_Variables,*numericVariables)
    
    ### Creating Train Data for model -------------------------------------
    X_train = Data.loc[:,YMC_Variables].astype(float)
    Y_train = Data['Target']
    
    ### Removing null from the model ---------------------
    X_train = X_train.loc[~np.isnan(Y_train),:].reset_index(drop=True)
    Y_train = Y_train.loc[~np.isnan(Y_train)].reset_index(drop=True)

    ### Defining the model ------------------------------------------------
    lgb_estimator = LightGBM.LGBMRegressor(boosting_type='gbdt',
                                            objective='regression')
    
    ### Defining the Grid -------------------------------------------------
    parameters = {'num_leaves':[20,40,60,80,100], 
                    'C': [0, 0.3, 0.5, 1],
                    'n_estimators': [50,100,150,200,300],
                    'min_child_samples':[5,10,15],
                    'max_depth':[-1,5,10,20,30,40,45],
                    'learning_rate':[0.05,0.1,0.2],
                    'reg_alpha':[0,0.01,0.03]}

    ### Run the model -----------------------------------------------------
    GBM_grid_search = RandomizedSearchCV(estimator = lgb_estimator,
                                            param_distributions = parameters,
                                            scoring='neg_mean_absolute_error',#'accuracy',,‘neg_mean_absolute_error’,'neg_root_mean_squared_error
                                            n_iter=30,
                                            cv = 4,
                                            n_jobs = 4)
    GBNModel = GBM_grid_search.fit(X=X_train, y=Y_train)   
    
    ### Fitting the best model --------------------------------------------
    GBNModel = GBNModel.best_estimator_.fit(X=X_train, y=Y_train)
        
    ### Saving the features name ------------------------------------------
    GBNModel.feature_names = list(YMC_Variables)
    
    del YMC_Variables
    del parameters
    del lgb_estimator
    del X_train
    del Y_train
    del GBM_grid_search

    ### Current Time ----------------------------------------------------------
    now = datetime.now() 
    CreateModelDate = now.strftime("%Y-%m-%d %H:%M:%S")

    ### Save The Model --------------------------------------------------------  
    Path = 'Model/Model.pckl'
    #os.getcwd()
    #Path = Path.replace('Segment', Segment, 1)  
    f = open(Path, 'wb')
    pickle.dump([YMC_Factor_Dictionary_List,
                totalYMeanTarget,
                totalYMedianTarget,
                YMC_Dictionary_Numeric_List,
                GBNModel,
                CreateModelDate], f)
    f.close()
