import pandas as pd
import numpy as np
#import statsmodels.formula.api as smf
#from sklearn import metrics
import os
import pickle
from datetime import datetime
#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
 
import lightgbm as LightGBM
#from sklearn.linear_model import LogisticRegression
#import lifelines
#from lifelines.utils import k_fold_cross_validation

#import pyarrow.parquet as pq
#from sklearn.linear_model import QuantileRegressor
#import shap
#import sqlalchemy

import Functions 

## ------------------------------------------------------------------------------------------------
## --------------------------------------- Read Data ----------------------------------------------
## ------------------------------------------------------------------------------------------------
import ReadData
Data = ReadData.Data

## Creating the Output Sample -------------------------------------------------
ModelOutput = pd.DataFrame()


## ------------------------------------------------------------------------------------------------
## ----------------------------------------- Model ------------------------------------------------
## ------------------------------------------------------------------------------------------------

def Model(Data,conf):
    ###Classifying Variables --------------------------------------------------
    FactorVariables = ["decision_no",
                        "t_claim_type", "t_profession", "postal_code", 
                        
                        "wrk_bck_dt_isnotnull","returnedbeforeregistration",
                        
                        "socio_economic_rate_points", "city","geo_location",
                        
                        "gender_id","marital_status", "smoke_ind", "domain", "occupation_name",
        
                        "first_dim_accident", "exception_str",
                        
                        "first_dim","first_sivug", "first_tat_sivug", "ind_ortopedia", 
                        "ind_cancer","ind_psych","ind_digesting","ind_women","ind_heart",
                        "ind_operation","ind_eyes","first_exception_str",
        
                        "past_suspicious",
                        
                        "ind_legal",
                        
                        "policy_count", "collective", "primary_cover_number",
                        "max_prat_mnahalim_ind",
                        "have_agent", "agent_is_active", "claim_agent_name",
                        "num_past_related_claims",
                        "primary_past_claim_desc", "second_primary_claim_desc",
                        "num_past_field_elementry", "num_past_field_health", "num_past_field_life",
    
                        "suplier_company_doctor",
                        "suplier_medical_card", "suplier_prior_investigation",
                        "suplier_investigation", "suplier_medical_commity", "suplier_legal",
                        "performer_company_doctor", "performer_medical_card",
                        "performer_prior_investigation", "performer_investigation",
                        "performer_medical_commity", "performer_legal",
                        
                        "decision_reject_ind","decision_approved_ind","special_pay_ind",
                        
                        
                        "related_sup_company_doctor",
                        "related_sup_medical_card", "related_sup_prior_invest",
                        "related_sup_investigation", "related_sup_medical_commity",
                        "related_sup_legal", "related_per_company_doctor",
                        "related_per_medical_card", "related_per_prior_invest",
                        "related_per_investigation", "related_per_medical_commity",
                        "related_per_legal"]

    NumericVariables = ["time_event_to_registration","time_registration_flag_date",
                        "age_at_event",
                        
                        "timediffreturnedwork",
                        
                        "socio_economic_rate_points",
                        "cover_duration_policy_months","max_waiting_time_for_premia",
                        "sum_cover_add_prof","sum_cover_add_md","tb_premia_registration",
                        "sum_cover_amount",
                        "sum_premia_policy","primary_cover_premia",
                            
                        "number_of_documents",
                        "num_must", "max_registration_doc_datediff", "num_doc_code_38",
                        "num_doc_code_4", "num_doc_code_16", "num_doc_code_100",
                        "num_doc_code_1", "num_doc_code_62", "num_doc_code_8",
                        "num_doc_code_30","num_doc_code_20", "num_doc_code_10","num_doc_code_25",
                        "num_doc_code_28","num_doc_code_42","num_doc_code_50",
                        "num_doc_code_138","num_doc_code_45","num_doc_code_11",
                        "num_doc_code_37","num_doc_code_13","num_doc_code_61",
                        
                        "number_of_letters",


                        "monthly_salary",
                        
                        "menora_codes_index",
                        
                        "num_past_related_claims",
                        "range_pasts_event", "min_time_between_past_event","max_time_between_past_event",
                        "proportion_of_past_payed","num_past_payed_related_events",
                        "num_past_field_elementry","num_past_field_health","num_past_field_life",
                        
                        "max_time_status_flagdate", "min_time_status_flagdate",
                        "time_first_descision", "time_open_closed_claim", "status_code_1",
                        "status_code_2", "status_code_3", "status_code_4", "status_code_5",
                        "status_code_6", "status_code_7", "station_desc_approved",
                        "station_desc_closed",

                        "num_supliers","supliers_payment",
                        
                        "num_decision",'max_dif_duration_payment',"time_duration_payment",
                        

                        "payed_money_until_flag_date",

                        
                        "num_past_claim_type_lwc", "num_past_claim_type_accidents",
                        "proportion_paid_past_life", "proportion_paid_past_elementry",
                        "proportion_paid_past_health", "num_past_green", 
                        
                        "sumpastrenewal", 
                        
                        "related_num_supliers",
                            
                        "num_paid_relaited_lwc_lwc",
                        "num_paid_relaited_lwc_accid", "num_paid_relaited_accid_lwc",
                        "num_paid_relaited_accid_accid", "sum_paid_relaited_lwc_lwc",
                        "sum_paid_relaited_lwc_accid", "sum_paid_relaited_accid_lwc",
                        "sum_paid_relaited_accid_accid"]
    DataModel.loc[:,NumericVariables] = DataModel.loc[:,NumericVariables].astype(float)
            
    ### Creating YMC Dictionaries for Factors (Creating the dictionaries) -----
    YMC_Factor_Dictionary_List = dict()
    for VariableToConvert in FactorVariables:
        # VariableToConvert="related_per_legal"
        # Creating variable to transform it to YMC ------------------------
        Variable = DataModel.loc[:, ["ClaimNo_Descision","percenthandicape", "y_suspicious","time_duration_days",VariableToConvert]]
        Variable.columns = ["ClaimNo_Descision","percenthandicape", "y_suspicious","time_duration_days", "VariableToConvert"]
        Variable.loc[:, 'VariableToConvert'] = Variable['VariableToConvert'].astype(str)
        Variable.loc[:, 'VariableToConvert'] = Variable['VariableToConvert'].fillna('NULL')        
    
        # Group all the Not Frequent Factor to one factor group -----------
        NotFrequentFactorGroup = pd.DataFrame(Variable.groupby('VariableToConvert')['percenthandicape'].apply(lambda x: 'Rare' if len(x) <= 15 else 'Frequent')).reset_index()
        NotFrequentFactorGroup.columns = ["VariableName", "SmallGroupOrNot"]
        FrequentFactors = NotFrequentFactorGroup.loc[NotFrequentFactorGroup.SmallGroupOrNot == 'Frequent'].VariableName
        Variable.loc[:, 'VariableToConvert'] = np.where(Variable['VariableToConvert'].isin(FrequentFactors), Variable['VariableToConvert'], 'Not Frequent Factor')
    
        # Creating Dictionary
        Dictionary_Variable_YMC = Variable.groupby('VariableToConvert')[["percenthandicape", "y_suspicious","time_duration_days"]].apply(np.mean).reset_index()
        Dictionary_Variable_YMC.columns = ["Variable","YMC_PercentHandicape_Mean_Variable","YMC_Suspicious_Mean_Variable","time_duration_days"]
        Dictionary_Variable_YMC = Dictionary_Variable_YMC.sort_values(by='YMC_PercentHandicape_Mean_Variable', ascending=False)

        Dictionary = pd.DataFrame(data = {"VariableToConvert": DataModel[VariableToConvert].unique()})
        Dictionary['VariableToConvert'] = Dictionary['VariableToConvert'].astype(str)
        Dictionary['VariableToConvert'] = Dictionary['VariableToConvert'].fillna('NULL') 
        Dictionary['Variable'] = np.where(Dictionary['VariableToConvert'].isin(FrequentFactors), Dictionary['VariableToConvert'], 'Not Frequent Factor')
        Dictionary = Dictionary.join(Dictionary_Variable_YMC.set_index('Variable'), how='left', on='Variable')
        Dictionary = Dictionary.drop(columns = 'Variable')
        Dictionary.columns = Dictionary_Variable_YMC.columns
        
        # Inserting the dictionary into a list
        YMC_Factor_Dictionary_List[VariableToConvert] = Dictionary
        
        
    ### Delete all temporary Variables ----------------------------------------
    del VariableToConvert
    del Variable
    del Dictionary_Variable_YMC
    del Dictionary
    del NotFrequentFactorGroup
    del FrequentFactors
        
    ### Inserting the Total YMC Measures for all the new predictions ----------
    TotalYMean_PercentHandicape = np.mean(DataModel['percenthandicape'])
    TotalYMean_Suspicious = np.mean(DataModel['y_suspicious'])   
    TotalYMean_TimeDuration = np.mean(DataModel['time_duration_days'])   

    
    ### Inserting the YMC Values from the dictionaries to the DataPanel -------
    for VariableName in YMC_Factor_Dictionary_List:
        # VariableName="num_past_field_life"
        DataModel.loc[:, VariableName] = DataModel[VariableName].astype(str)
        
        YMC_Dictionary = YMC_Factor_Dictionary_List[VariableName]
        YMC_Dictionary.columns = [VariableName, 
                                    VariableName+"_Mean_YMC_PercentHandicape",
                                    VariableName+"_Mean_YMC_Suspicious",
                                    VariableName+"_Mean_YMC_TimeDuration"]
    
        DataModel = DataModel.join(YMC_Dictionary.set_index([VariableName]), how='left', on=[VariableName])
        DataModel.loc[:, VariableName+"_Mean_YMC_PercentHandicape"] = DataModel[VariableName+"_Mean_YMC_PercentHandicape"].fillna(TotalYMean_PercentHandicape)        
        DataModel.loc[:, VariableName+"_Mean_YMC_Suspicious"] = DataModel[VariableName+"_Mean_YMC_Suspicious"].fillna(TotalYMean_Suspicious)
        DataModel.loc[:, VariableName+"_Mean_YMC_TimeDuration"] = DataModel[VariableName+"_Mean_YMC_TimeDuration"].fillna(TotalYMean_TimeDuration)
    
    ### Delete all temporary Variables ----------------------------------------
    del YMC_Dictionary
    del VariableName 
        
    ### Numerical Data Manipulation (YMC) -------------------------------------
    YMC_Dictionary_Numeric_List = dict()
    for VariableToConvert in NumericVariables:
        Variable = DataModel[VariableToConvert].astype(float)
        Variable = Variable.fillna(0)
        YMC_Dictionary_Numeric_List[VariableToConvert] = MultipleNumericYMC(Variable = Variable,
                                                                            PercentHandicape = DataModel['percenthandicape'],
                                                                            Suspicious = DataModel['y_suspicious'],
                                                                            TimeDurationDays = DataModel['time_duration_days'],
                                                                            NumberOfGroups = max(10, round(len(DataModel) / 150)))
    
    ### Creating the YMC calculation for each numeric variable ----------------
    Numeric_YMC = pd.DataFrame(data={'ClaimNo_Descision': DataModel['ClaimNo_Descision']})
    for VariableToConvert in NumericVariables:
        Variable = pd.DataFrame(data={'ClaimNo_Descision': DataModel['ClaimNo_Descision'], 
                                        VariableToConvert: DataModel[VariableToConvert].astype(float)})
        Variable.loc[:,VariableToConvert] = Variable[VariableToConvert].fillna(0)
    
        # Inserting the numeric dictionary into VariableDictionary
        VariableDictionary = YMC_Dictionary_Numeric_List[VariableToConvert]
    
        # Adding All the YMC
        VariableDictionary.index = pd.IntervalIndex.from_arrays(VariableDictionary['lag_value'],
                                                                VariableDictionary['value'],
                                                                closed='left')
        V = Variable[VariableToConvert]
        Variable[['Mean_YMC_PercentHandicape','Mean_YMC_YSuspicious', 'Mean_YMC_TimeDurationDays']] = VariableDictionary.loc[V][['Mean_YMC_PercentHandicape', 'Mean_YMC_YSuspicious', 'Mean_YMC_TimeDurationDays']].reset_index(drop=True)
    
        # Creating YMC table
        YMC = pd.DataFrame(data={'VariableToConvert_Numeric_Mean_YMC_PercentHandicape': Variable['Mean_YMC_PercentHandicape'],                                     
                                    'VariableToConvert_Numeric_Mean_YMC_Suspicious': Variable['Mean_YMC_YSuspicious'],
                                    'VariableToConvert_Numeric_Mean_YMC_TimeDurationDays': Variable['Mean_YMC_TimeDurationDays']})
        
        # Left join YMC table to NUmeric_YMC table
        Numeric_YMC = pd.concat([Numeric_YMC, YMC], axis=1)
        Numeric_YMC.columns = list(map(lambda x: x.replace('VariableToConvert', VariableToConvert, 1), Numeric_YMC.columns))
    
    ### Left join of Numeric_YMC table to the DataPanel -----------------------
    DataModel = DataModel.join(Numeric_YMC.set_index('ClaimNo_Descision'), how='left', on='ClaimNo_Descision')
    
    ### Delete all temporary Variables ----------------------------------------
    del VariableToConvert
    del Numeric_YMC 
    del Variable
    del VariableDictionary
    del V
    del YMC
    
    
    ### -----------------------------------------------------------------------
    ### ------------------- Suspicious Model ----------------------------------
    ### -----------------------------------------------------------------------
    ### Taking the YMC_Suspicious variables -----------------------------------
    YMC_Suspicious = DataModel.columns[["_YMC_Suspicious" in i for i in DataModel.columns]]

    ### Logistic Regression Model - for Suspicious ----------------------------
    try:
        # Removing the correlated variables
        TempData = DataModel.loc[:, YMC_Suspicious].astype(float)
        TempData = TempData.iloc[:, ~(TempData.apply(lambda x: round(np.var(x.astype(float)),4) == 0,axis=0).values)]  
        CorMat = TempData.corr().abs()# Create correlation matrix 
        CorMat.iloc[np.triu(np.ones(CorMat.shape), k=0).astype(bool)] = 0
        to_drop = [column for column in CorMat.columns if any(CorMat[column] > 0.80)]
        TempData = TempData.drop(TempData[to_drop], axis=1)# Drop features  
        VariablesToTheModel_Suspicious  = TempData.columns##colnames of variables that don't correlated
    
        # Creating the formula
        if ClaimType=='Accidental Hendicapt':
            SuspiciousForm = "+".join(["y_suspicious~1",*VariablesToTheModel_Suspicious])
        else:
            SuspiciousForm = "+".join(["y_suspicious~1",*VariablesToTheModel_Suspicious])

        # Run the model
        SuspiciousModel = smf.logit(SuspiciousForm, DataModel)
        SuspiciousModel = SuspiciousModel.fit_regularized(maxiter=1000)
    except: 
        try:
            # Removing the correlated variables
            TempData = DataModel.loc[:, YMC_Suspicious].astype(float)
            TempData = TempData.iloc[:, ~(TempData.apply(lambda x: round(np.var(x.astype(float)),4) == 0,axis=0).values)]  
            CorMat = TempData.corr().abs()# Create correlation matrix 
            CorMat.iloc[np.triu(np.ones(CorMat.shape), k=0).astype(bool)] = 0
            to_drop = [column for column in CorMat.columns if any(CorMat[column] > 0.7)]
            TempData = TempData.drop(TempData[to_drop], axis=1)# Drop features  
            VariablesToTheModel_Suspicious  = TempData.columns##colnames of variables that don't correlated
        
            # Creating the formula
            if ClaimType=='Accidental Hendicapt':
                SuspiciousForm = "+".join(["y_suspicious~1",*VariablesToTheModel_Suspicious])
            else:
                SuspiciousForm = "+".join(["y_suspicious~1",*VariablesToTheModel_Suspicious])
            
            # Run the model
            SuspiciousModel = smf.logit(SuspiciousForm, DataModel)
            SuspiciousModel = SuspiciousModel.fit(maxiter=100)
        except:
            # Removing the correlated variables
            TempData = DataModel.loc[:, YMC_Suspicious].astype(float)
            TempData = TempData.iloc[:, ~(TempData.apply(lambda x: round(np.var(x.astype(float)),4) == 0,axis=0).values)]  
            CorMat = TempData.corr().abs()# Create correlation matrix 
            CorMat.iloc[np.triu(np.ones(CorMat.shape), k=0).astype(bool)] = 0
            to_drop = [column for column in CorMat.columns if any(CorMat[column] > 0.2)]
            TempData = TempData.drop(TempData[to_drop], axis=1)# Drop features  
            VariablesToTheModel_Suspicious  = TempData.columns##colnames of variables that don't correlated
        
            # Creating the formula
            if ClaimType=='Accidental Hendicapt':
                SuspiciousForm = "+".join(["y_suspicious~1",*VariablesToTheModel_Suspicious])
            else:
                SuspiciousForm = "+".join(["y_suspicious~1",*VariablesToTheModel_Suspicious])
            
            # Run the model
            SuspiciousModel = smf.logit(SuspiciousForm, DataModel)
            SuspiciousModel = SuspiciousModel.fit()

    ### Predictions -----------------------------------------------------------
    DataModel['Predicted_Suspicious'] = SuspiciousModel.predict(DataModel)
    
    ### Check the model -------------------------------------------------------
    #print(SuspiciousModel.summary())
    #DataModel.groupby('y_suspicious')['Predicted_Suspicious'].mean().reset_index()
    
    ### Saving the features name ----------------------------------------------
    SuspiciousModel.feature_names = list(VariablesToTheModel_Suspicious)
    
    ### Shap - Variables Explanation of the model -----------------------------
    ModelForVariablesExplanation = LogisticRegression(max_iter=1000).fit(DataModel.loc[:,VariablesToTheModel_Suspicious].astype(float), DataModel['y_suspicious'])     
    ModelForVariablesExplanation.feature_names = list(VariablesToTheModel_Suspicious)

    ### Delete all temporary Variables ----------------------------------------
    del VariablesToTheModel_Suspicious
    del SuspiciousForm
    del YMC_Suspicious
    del CorMat
    del to_drop
    del TempData
    
    ### -----------------------------------------------------------------------
    ### ------------------------ Percent Handicape ----------------------------
    ### -----------------------------------------------------------------------
    if Segment in ['ZeroModel - Accidental Hendicapt - 1','FirstUpdate - Accidental Hendicapt - 1','OnGoingModel - Accidental Hendicapt - 1',
                    'ZeroModel - Accidental Hendicapt - 2','FirstUpdate - Accidental Hendicapt - 2','OnGoingModel - Accidental Hendicapt - 2',
                    'ZeroModel - Accidental Hendicapt - 3','FirstUpdate - Accidental Hendicapt - 3','OnGoingModel - Accidental Hendicapt - 3',
                    'ZeroModel - Accidental Hendicapt - 4','FirstUpdate - Accidental Hendicapt - 4','OnGoingModel - Accidental Hendicapt - 4',
                    'ZeroModel - Accidental Hendicapt - 5','FirstUpdate - Accidental Hendicapt - 5','OnGoingModel - Accidental Hendicapt - 5',
                    'ZeroModel - Accidental Hendicapt - 6','FirstUpdate - Accidental Hendicapt - 6','OnGoingModel - Accidental Hendicapt - 6']:
        ### Taking the variables to the model----------------------------------
        GBM_Variables_PercentHandicape = DataModel.columns[["_YMC_PercentHandicape" in i for i in DataModel.columns]]
        GBM_Variables_PercentHandicape = (*GBM_Variables_PercentHandicape,*NumericVariables)
        
        ### Creating Train Data for model -------------------------------------
        X_train = DataModel.loc[:,GBM_Variables_PercentHandicape].astype(float)
        Y_train = DataModel['percenthandicape']
        
        ### Removing percenthandicape null from the model ---------------------
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
                                                n_iter=130,
                                                cv = 4,
                                                n_jobs = 4)
        PercentHandicape_GBM = GBM_grid_search.fit(X=X_train, y=Y_train)   
    
        ### Fitting the best model --------------------------------------------
        PercentHandicape_GBM = PercentHandicape_GBM.best_estimator_.fit(X=X_train, y=Y_train)
        
        ### Saving the features name ------------------------------------------
        PercentHandicape_GBM.feature_names = list(GBM_Variables_PercentHandicape)
    
        del GBM_Variables_PercentHandicape
        del parameters
        del lgb_estimator
        del X_train
        del Y_train
        del GBM_grid_search
    else:
        PercentHandicape_GBM = 1
    
    ### -----------------------------------------------------------------------
    ### ----------------------- More Than Two Years ---------------------------
    ### -----------------------------------------------------------------------
    ### Taking the variables to the model--------------------------------------
    if Segment in ['ZeroModel - Lost Working Abbilities - 1','FirstUpdate - Lost Working Abbilities - 1','OnGoingModel - Lost Working Abbilities - 1',
                    'ZeroModel - Lost Working Abbilities - 2','FirstUpdate - Lost Working Abbilities - 2','OnGoingModel - Lost Working Abbilities - 2',
                    'ZeroModel - Lost Working Abbilities - 3','FirstUpdate - Lost Working Abbilities - 3','OnGoingModel - Lost Working Abbilities - 3',
                    'ZeroModel - Lost Working Abbilities - 4','FirstUpdate - Lost Working Abbilities - 4','OnGoingModel - Lost Working Abbilities - 4',
                    'ZeroModel - Lost Working Abbilities - 5','FirstUpdate - Lost Working Abbilities - 5','OnGoingModel - Lost Working Abbilities - 5',
                    'ZeroModel - Lost Working Abbilities - 6','FirstUpdate - Lost Working Abbilities - 6','OnGoingModel - Lost Working Abbilities - 6']:
        MoreThanTwoYears = DataModel.columns[["_YMC_TimeDurationDays" in i for i in DataModel.columns]]
            
        ### Removing the correlated variables ---------------------------------
        TempData = DataModel.loc[DataModel['wrk_bck_dt'].isna(),:]
        TempData = TempData.loc[:, MoreThanTwoYears].astype(float)
        TempData = TempData.iloc[:, ~(TempData.apply(lambda x: round(np.var(x.astype(float)),4) == 0,axis=0).values)]  
        CorMat = TempData.corr().abs()# Create correlation matrix 
        CorMat.iloc[np.triu(np.ones(CorMat.shape), k=0).astype(bool)] = 0
        to_drop = [column for column in CorMat.columns if any(CorMat[column] > 0.8)]
        TempData = TempData.drop(TempData[to_drop], axis=1)# Drop features  
        VariablesToTheModel_MoreThanTwoYears  = np.array(TempData.columns)##colnames of variables that don't correlated
        
        form = "+".join(VariablesToTheModel_MoreThanTwoYears)
    
        ### Cox Model ---------------------------------------------------------
        CoxModelData = DataModel.loc[DataModel['wrk_bck_dt'].isna(),:]
        CoxModelData['Y_Survival'] = CoxModelData['time_duration_days'].fillna(0)  
        CoxModelData['status'] = (1-CoxModelData['censoredornot'])    
    
        ### Sampled Data (Bootsrap) ------------------------------------------- 
        try:
            cph = lifelines.CoxPHFitter(penalizer=0.1)
            CoxModel = cph.fit(CoxModelData.loc[:,['Y_Survival','status',*VariablesToTheModel_MoreThanTwoYears]],
                                duration_col='Y_Survival',event_col='status',
                                formula=form,show_progress=False)#Concordance = 0.53
            CoxModel.concordance_index_##0.5307184998497475
        except:
            CollinearitySolutionTable = CoxModelData.loc[:, ["_YMC_TimeDurationDays" in i for i in CoxModelData.columns]]
            #CollinearitySolutionTable = CoxModelData.loc[:, VariablesToTheModel_MoreThanTwoYears].astype(float)
            CollinearitySolutionTable = CollinearitySolutionTable + np.random.uniform(-0.00001, 0.00001, np.array(CollinearitySolutionTable.shape))
            CollinearitySolutionTable = CollinearitySolutionTable + np.random.uniform(-0.00001, 0.00001, np.array(CollinearitySolutionTable.shape))
            CollinearitySolutionTable = CollinearitySolutionTable + np.random.uniform(-0.00001, 0.00001, np.array(CollinearitySolutionTable.shape))
            CollinearitySolutionTable = CollinearitySolutionTable + np.random.uniform(-0.00001, 0.00001, np.array(CollinearitySolutionTable.shape))
            CollinearitySolutionTable = CollinearitySolutionTable + np.random.uniform(-0.00001, 0.00001, np.array(CollinearitySolutionTable.shape))
            CollinearitySolutionTable = CollinearitySolutionTable + np.random.uniform(-0.0001, 0.0001, np.array(CollinearitySolutionTable.shape))
            CollinearitySolutionTable = CollinearitySolutionTable + np.random.uniform(-0.0001, 0.0001, np.array(CollinearitySolutionTable.shape))
            CollinearitySolutionTable = CollinearitySolutionTable + np.random.uniform(-0.0001, 0.0001, np.array(CollinearitySolutionTable.shape))
            CollinearitySolutionTable = CollinearitySolutionTable + np.random.uniform(-0.0001, 0.0001, np.array(CollinearitySolutionTable.shape))
            CollinearitySolutionTable = CollinearitySolutionTable + np.random.uniform(-0.0001, 0.0001, np.array(CollinearitySolutionTable.shape))
            CollinearitySolutionTable = CollinearitySolutionTable + np.random.uniform(-0.001, 0.001, np.array(CollinearitySolutionTable.shape))
            CollinearitySolutionTable = CollinearitySolutionTable + np.random.uniform(-0.001, 0.001, np.array(CollinearitySolutionTable.shape))
            CollinearitySolutionTable['Y_Survival'] = CoxModelData['Y_Survival']
            CollinearitySolutionTable['status'] = CoxModelData['status']
    
            cph = lifelines.CoxPHFitter(penalizer=0.1)
            CoxModel = cph.fit(CollinearitySolutionTable, duration_col='Y_Survival', event_col='status',formula=form, show_progress=False, step_size=0.01)
    
    
        ### Reducing the Model ------------------------------------------------
        ReducedModel = pd.DataFrame(data={'Variable': CoxModel.summary.index.values,
                                            'PValue': CoxModel.summary['p'].values.astype(float)})
        ReducedModel=ReducedModel.loc[ReducedModel['PValue']<=0.6,:]
        
        ### New Formula -------------------------------------------------------
        NewForm = ReducedModel['Variable'].values
        NewForm = "+".join(NewForm)
    
        
        try:
            cph = lifelines.CoxPHFitter()
            CoxModel = cph.fit(CoxModelData.loc[:,['Y_Survival','status',*ReducedModel['Variable'].values]], 
                                duration_col='Y_Survival',
                                event_col='status',
                                formula=NewForm,
                                show_progress=False)
            CoxModel.concordance_index_##0.5307752533098541
        except:
            CollinearitySolutionTable = CoxModelData.loc[:, ["_YMC" in i for i in CoxModelData.columns]]
            #CollinearitySolutionTable = CoxModelData.loc[:, VariablesToTheModel_MoreThanTwoYears].astype(float)
            CollinearitySolutionTable = CollinearitySolutionTable + np.random.uniform(-0.00001, 0.00001, np.array(CollinearitySolutionTable.shape))
            CollinearitySolutionTable = CollinearitySolutionTable + np.random.uniform(-0.00001, 0.00001, np.array(CollinearitySolutionTable.shape))
            CollinearitySolutionTable = CollinearitySolutionTable + np.random.uniform(-0.00001, 0.00001, np.array(CollinearitySolutionTable.shape))
            CollinearitySolutionTable = CollinearitySolutionTable + np.random.uniform(-0.00001, 0.00001, np.array(CollinearitySolutionTable.shape))
            CollinearitySolutionTable = CollinearitySolutionTable + np.random.uniform(-0.00001, 0.00001, np.array(CollinearitySolutionTable.shape))
            CollinearitySolutionTable = CollinearitySolutionTable + np.random.uniform(-0.0001, 0.0001, np.array(CollinearitySolutionTable.shape))
            CollinearitySolutionTable = CollinearitySolutionTable + np.random.uniform(-0.0001, 0.0001, np.array(CollinearitySolutionTable.shape))
            CollinearitySolutionTable = CollinearitySolutionTable + np.random.uniform(-0.0001, 0.0001, np.array(CollinearitySolutionTable.shape))
            CollinearitySolutionTable = CollinearitySolutionTable + np.random.uniform(-0.0001, 0.0001, np.array(CollinearitySolutionTable.shape))
            CollinearitySolutionTable = CollinearitySolutionTable + np.random.uniform(-0.0001, 0.0001, np.array(CollinearitySolutionTable.shape))
            CollinearitySolutionTable = CollinearitySolutionTable + np.random.uniform(-0.001, 0.001, np.array(CollinearitySolutionTable.shape))
            CollinearitySolutionTable = CollinearitySolutionTable + np.random.uniform(-0.001, 0.001, np.array(CollinearitySolutionTable.shape))
            CollinearitySolutionTable['Y_Survival'] = CoxModelData['Y_Survival']
            CollinearitySolutionTable['status'] = CoxModelData['status']
    
            cph = lifelines.CoxPHFitter(penalizer=0.1)
            CoxModel = cph.fit(CollinearitySolutionTable, duration_col='Y_Survival', event_col='status',formula=NewForm, show_progress=False, step_size=0.01)
            
        ### Cross validation --------------------------------------------------
        try:          
            k_fold_cross_validation(fitters = CoxModel, 
                                    df = CoxModelData.loc[:,['Y_Survival','status',*ReducedModel['Variable'].values]].reset_index(drop=True),
                                    duration_col='Y_Survival', 
                                    event_col='status',
                                    k=2, 
                                    scoring_method="concordance_index")
        except:
            1+1
        
        
        ### Model Summery -----------------------------------------------------
        # ModelSummery = pd.DataFrame(data={'Variable': CoxModel.summary.index.values,
        #                    'tValues': CoxModel.summary['z'].values.astype(float)})
        # ModelSummery['Variable'] = list(map(lambda x: x.replace('_Mean_YMC_TimeDuration','').replace('_Median_YMC_TimeDuration','').replace('_Quantile99_YMC_TimeDuration','').replace('_Sd_YMC_TimeDuration','').replace('_Numeric',''), ModelSummery['Variable']))
        # Dictionary = pd.read_excel('Menora/Variable Dictionary.xlsx')
        # ModelSummery["Variable"].replace(dict(Dictionary.values), inplace=True)
        
        ### Delete all temporary Variables ----------------------------------------
        del VariablesToTheModel_MoreThanTwoYears
        del cph
        del MoreThanTwoYears
        del CorMat
        del to_drop
        del TempData
        del form
        del CoxModelData
    
    else:
        CoxModel = 1 
    
    ### Current Time ----------------------------------------------------------
    now = datetime.now() 
    CreateModelDate = now.strftime("%Y-%m-%d %H:%M:%S")
    #print("Current Time =", CreateModelDate)
    
    ## Save The Data Panels ---------------------------------------------------
    Path = 'Menora/DataPanels/DataModel - Segment.pckl'
    Path = Path.replace('Segment', Segment, 1)  
    DataModel.to_csv(Path,encoding='utf-8-sig')
    
    
    ### Suspicious - Predictions ----------------------------------------------
    DataModel['Predicted_Suspicious'] = SuspiciousModel.predict(DataModel)
    
    
    # Percent Handicape - GBM Model - Predictions------------------------------
    if Segment in ['ZeroModel - Accidental Hendicapt - 1','FirstUpdate - Accidental Hendicapt - 1','OnGoingModel - Accidental Hendicapt - 1',
                    'ZeroModel - Accidental Hendicapt - 2','FirstUpdate - Accidental Hendicapt - 2','OnGoingModel - Accidental Hendicapt - 2',
                    'ZeroModel - Accidental Hendicapt - 3','FirstUpdate - Accidental Hendicapt - 3','OnGoingModel - Accidental Hendicapt - 3',
                    'ZeroModel - Accidental Hendicapt - 4','FirstUpdate - Accidental Hendicapt - 4','OnGoingModel - Accidental Hendicapt - 4',
                    'ZeroModel - Accidental Hendicapt - 5','FirstUpdate - Accidental Hendicapt - 5','OnGoingModel - Accidental Hendicapt - 5',
                    'ZeroModel - Accidental Hendicapt - 6','FirstUpdate - Accidental Hendicapt - 6','OnGoingModel - Accidental Hendicapt - 6']:
        DataModel['PredictedPercentHandicape_GBM'] = PercentHandicape_GBM.predict(DataModel.loc[:,PercentHandicape_GBM.feature_names].astype(float))
        DataModel['PredictedPercentHandicape_GBM'] = DataModel['PredictedPercentHandicape_GBM'].astype(float)
        DataModel['PredictedPercentHandicape_GBM'] = np.where(DataModel['PredictedPercentHandicape_GBM']<=0,0,DataModel['PredictedPercentHandicape_GBM'])
        DataModel['PredictedPercentHandicape_GBM'] = np.where(DataModel['PredictedPercentHandicape_GBM']>=100,100,DataModel['PredictedPercentHandicape_GBM'])
    else:
        DataModel['PredictedPercentHandicape_GBM'] = np.nan
        

    ## More Than two Years - Predictions --------------------------------------
    #### Regular Predictions --------------------------------------------------
    #Cox model Prediction
    if Segment in ['ZeroModel - Lost Working Abbilities - 1','FirstUpdate - Lost Working Abbilities - 1','OnGoingModel - Lost Working Abbilities - 1',
                    'ZeroModel - Lost Working Abbilities - 2','FirstUpdate - Lost Working Abbilities - 2','OnGoingModel - Lost Working Abbilities - 2',
                    'ZeroModel - Lost Working Abbilities - 3','FirstUpdate - Lost Working Abbilities - 3','OnGoingModel - Lost Working Abbilities - 3',
                    'ZeroModel - Lost Working Abbilities - 4','FirstUpdate - Lost Working Abbilities - 4','OnGoingModel - Lost Working Abbilities - 4',
                    'ZeroModel - Lost Working Abbilities - 5','FirstUpdate - Lost Working Abbilities - 5','OnGoingModel - Lost Working Abbilities - 5',
                    'ZeroModel - Lost Working Abbilities - 6','FirstUpdate - Lost Working Abbilities - 6','OnGoingModel - Lost Working Abbilities - 6']:
        Probabilities = 1 - pd.DataFrame(CoxModel.predict_survival_function(DataModel, [10,30,60,90,180,360,730])).T
        Probabilities.columns = list(map(lambda x: 'Prob' + str(int(x)), [10,30,60,90,180,360,730]))
        #### Merge to the DataPanel -------------------------------------------
        DataModel = pd.concat([DataModel.reset_index(drop=True), Probabilities.reset_index(drop=True)], axis=1)
        
        ## If The Claimer already returned to work we will manually change his prediction
        DataModel['Prob90'] = np.where(DataModel['timediffreturnedwork']>=90,1,DataModel['Prob90'])
        DataModel['Prob180'] = np.where(DataModel['timediffreturnedwork']>=180,1,DataModel['Prob180'])
        DataModel['Prob360'] = np.where(DataModel['timediffreturnedwork']>=360,1,DataModel['Prob360'])
        DataModel['Prob730'] = np.where(DataModel['timediffreturnedwork']>=730,1,DataModel['Prob730'])
        
        DataModel['Prob90'] = np.where(DataModel['returnedbeforeregistration'].astype(int)==1,1,DataModel['Prob90'])
        DataModel['Prob180'] = np.where(DataModel['returnedbeforeregistration'].astype(int)==1,1,DataModel['Prob180'])
        DataModel['Prob360'] = np.where(DataModel['returnedbeforeregistration'].astype(int)==1,1,DataModel['Prob360'])
        DataModel['Prob730'] = np.where(DataModel['returnedbeforeregistration'].astype(int)==1,1,DataModel['Prob730'])
    else:
        DataModel['Prob90'] = np.nan
        DataModel['Prob180'] = np.nan
        DataModel['Prob360'] = np.nan
        DataModel['Prob730'] = np.nan
        
        
    ### Model Output ----------------------------------------------------------------
    ModelOut = pd.DataFrame(data={'ClaimNo': DataModel['claim_no'],
                                    'RandSample': DataModel['RandSample'],
                                    'Segment': DataModel['Segment'],
                                    'Decision_no': DataModel['decision_no'],
                                    'ClaimType': DataModel['t_claim_type'],
                                    'Suspicious': DataModel['y_suspicious'],
                                    'Suspicious_Predicted': DataModel['Predicted_Suspicious'],                                
                                    'TotalPayment': DataModel['total_payment'],
                                    'TimeDurationDays_Prob90': DataModel['Prob90'],
                                    'TimeDurationDays_Prob180': DataModel['Prob180'],
                                    'TimeDurationDays_Prob360': DataModel['Prob360'],
                                    'TimeDurationDays_Prob730': DataModel['Prob730'],
                                    'TimeDurationDays': DataModel['time_duration_days'],
                                    'PredictedPercentHandicape_GBM': DataModel['PredictedPercentHandicape_GBM'],
                                    'PercentHandicape': DataModel['percenthandicape'],
                                    'LegalOrNoDecision': np.where(np.isnan(DataModel['percenthandicape']),1,0),
                                    'TimeDiffReturnedWork': DataModel['timediffreturnedwork'],
                                    'returnedbeforeregistration':DataModel['returnedbeforeregistration']
                                })
                                

    ModelOutput = pd.concat([ModelOutput,ModelOut])
    
    ### Save The Model --------------------------------------------------------  
    Path = 'Menora/Dynamic Models - Pickle/Model Checking - Segment.pckl'
    Path = Path.replace('Segment', Segment, 1)  
    f = open(Path, 'wb')
    pickle.dump([YMC_Factor_Dictionary_List,
                    TotalYMean_PercentHandicape,
                    TotalYMean_Suspicious,
                    TotalYMean_TimeDuration,
                    SuspiciousModel,
                    ModelForVariablesExplanation,
                    YMC_Dictionary_Numeric_List,
                    PercentHandicape_GBM,
                    CoxModel,
                    CreateModelDate], f)
    f.close()
    del f,Path,DataModel,CreateModelDate,now
    
