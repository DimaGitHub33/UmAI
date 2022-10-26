import pandas as pd
import numpy as np
# import statsmodels.formula.api as smf
# #from sklearn import metrics
# import os
# import pickle
# from datetime import datetime
# #from sklearn.ensemble import GradientBoostingClassifier
# #from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RandomizedSearchCV
 
# import lightgbm as LightGBM
# from sklearn.linear_model import LogisticRegression
# import lifelines
# from lifelines.utils import k_fold_cross_validation

# #from sklearn.linear_model import QuantileRegressor
# import shap
# import sqlalchemy

## ------------------------------------------------------------------------------------------------
## ----------------------------------------- Functions --------------------------------------------
## ------------------------------------------------------------------------------------------------
## Ranks Dictionary ----------------------------------------------------------
"""    
Ranks Dictionary
    Args:
        temp_data:
        Array of numerical data
        ranks_num:
        Number of ranks to split the temp_data

    Returns:
        pandas data frame that describe the boundaries of ranks for input data 
        Example:
        Dictionary = Ranks_Dictionary(np.random.normal(3, 2.5, size=(1, 1000)), ranks_num=10)
        print(Dictionary)
        quantile     value  lag_value  rank
                0.1 -0.210117       -inf     1
                0.2  0.748485  -0.210117     2
                0.3  1.636300   0.748485     3
                0.4  2.210183   1.636300     4
                0.5  2.832312   2.210183     5
                0.6  3.501649   2.832312     6
                0.7  4.116645   3.501649     7
                0.8  4.895207   4.116645     8
                0.9  5.843564   4.895207     9
                1.0       inf   5.843564    10

        The output table help us to split any numerical array to array of ranks.
        Each numerical value is between value and lag_value in this table. the rank in that specific row is the rank of the value.
"""

def Ranks_Dictionary(temp_data, ranks_num):
    quantile_seq = np.linspace(1 / ranks_num, 1, ranks_num)
    overall_quantile = list(map(lambda x: round(np.quantile(temp_data, x), 6), quantile_seq))
    overall_quantile = pd.concat([pd.DataFrame(quantile_seq), pd.DataFrame(overall_quantile)], axis=1)
    overall_quantile.columns = ['quantile', 'value']
    overall_quantile['lag_value'] = overall_quantile['value'].shift(1)
    overall_quantile.loc[:, 'lag_value'] = overall_quantile['lag_value'].fillna(float('-inf'))
    overall_quantile.loc[:, 'value'][len(overall_quantile['value']) - 1] = float('inf')
    overall_quantile['rank'] = list(range(1, len(overall_quantile['value']) + 1))
    overall_quantile = overall_quantile.loc[overall_quantile['value'] != overall_quantile['lag_value'], :]
    return overall_quantile

## jitter ---------------------------------------------------------------------
"""    
RJitter
    Args:
        x:
        Array of numerical data
        factor:
        rate of jitter

    Returns:
        array of numerical data with almost the same data (the mean is the same the variance slightly grows, depend on the factor)
        Example:
        InputData=np.random.normal(0, 2.5, size=(1, 3))[0]
        print(InputData)
        [-1.58334198  0.64810107  0.37624609]
        print(RJitter(x=InputData,factor=0.1))
        [-1.58345746  0.64537898  0.37463414]

        The function slightly add variance to the input data so that we can split it, if there are to many unique values.
"""

def RJitter(x, factor):
    z = max(x) - min(x)
    amount = factor * (z / 50)
    x = x + np.random.uniform(-amount, amount, len(x))
    return (x)

## Multiple Numeric YMC -------------------------------------------------------
def MultipleNumericYMC(Variable, PercentHandicape, Suspicious, TimeDurationDays, NumberOfGroups):

    # Create dictionary for Variable
    #Dictionary = Ranks_Dictionary(Variable + np.random.uniform(0, 0.00001, len(Variable)), ranks_num=NumberOfGroups)
    Dictionary = Ranks_Dictionary(RJitter(Variable,0.00001), ranks_num=NumberOfGroups)
    Dictionary.index = pd.IntervalIndex.from_arrays(Dictionary['lag_value'],
                                                    Dictionary['value'],
                                                    closed='left')
    # Convert Each value in variable to rank
    Variable = pd.DataFrame({'Variable': Variable, 
                             'Suspicious': Suspicious,
                             'PercentHandicape': PercentHandicape,
                             'TimeDurationDays': TimeDurationDays})
    V = Variable['Variable']
    Variable['rank'] = Dictionary.loc[V]['rank'].reset_index(drop=True)

    # Create The aggregation for each rank
    def aggregations(x):
        Mean_YMC = np.mean(x)
        NumberOfObservation = len(x)
        DataReturned = pd.DataFrame({'Mean_YMC_PercentHandicape': [Mean_YMC],
                                     'NumberOfObservation': [NumberOfObservation]})

        return DataReturned

    # Aggregation Table
    AggregationTable = Variable.groupby('rank')['PercentHandicape'].apply(aggregations).reset_index()
    AggregationTable = AggregationTable.drop(columns=['level_1'])

    AggregationTable_YSuspicious = Variable.groupby('rank')['Suspicious'].mean().reset_index('rank')
    AggregationTable_YSuspicious.columns = ['rank' ,'Mean_YMC_YSuspicious']
    
    AggregationTable_TimeDurationDays = Variable.groupby('rank')['TimeDurationDays'].mean().reset_index()
    AggregationTable_TimeDurationDays.columns = ['rank' ,'Mean_YMC_TimeDurationDays']

    # Join Aggregation Tables
    AggregationTable = AggregationTable.join(AggregationTable_YSuspicious.set_index('rank'), how='left', on='rank')
    AggregationTable = AggregationTable.join(AggregationTable_TimeDurationDays.set_index('rank'),how='left', on='rank')

    # Merge to The Dictionary
    Dictionary = Dictionary.merge(AggregationTable, how='left', on=['rank'])

    Dictionary.loc[:, 'Mean_YMC_PercentHandicape'] = Dictionary['Mean_YMC_PercentHandicape'].fillna(np.mean(PercentHandicape.dropna()))
    Dictionary.loc[:, 'NumberOfObservation'] = Dictionary['NumberOfObservation'].fillna(0)
    Dictionary.loc[:, 'Mean_YMC_YSuspicious'] = Dictionary['Mean_YMC_YSuspicious'].fillna(np.mean(Suspicious.dropna()))
    Dictionary.loc[:, 'Mean_YMC_TimeDurationDays'] = Dictionary['Mean_YMC_TimeDurationDays'].fillna(np.mean(TimeDurationDays.dropna()))

    return Dictionary



##  FourBasicNumericYMC -------------------------------------------------------
def FourBasicNumericYMC(Variable, Target, NumberOfGroups):
    # Create dictionary for Variable
    #Dictionary = Ranks_Dictionary(Variable + np.random.uniform(0, 0.00001, len(Variable)), ranks_num=NumberOfGroups)
    Dictionary = Ranks_Dictionary(RJitter(Variable,0.00001), ranks_num=NumberOfGroups)
    Dictionary.index = pd.IntervalIndex.from_arrays(Dictionary['lag_value'],
                                                    Dictionary['value'],
                                                    closed='left')
    # Convert Each value in variable to rank
    Variable = pd.DataFrame({'Variable': Variable, 
                             'Target': Target})
    IntervalLocation = Variable['Variable']
    Variable['rank'] = Dictionary.loc[IntervalLocation]['rank'].reset_index(drop=True)
    del IntervalLocation


    # Aggregation Table
    AggregationTable = pd.DataFrame()
    AggregationTable['rank'] = Variable['rank'].unique()
    AggregationTable = AggregationTable.merge(Variable.groupby('rank')['Target'].mean().reset_index().set_axis(['rank', 'Mean'], axis=1), how='left', on=['rank'])
    AggregationTable = AggregationTable.merge(Variable.groupby('rank')['Target'].median().reset_index().set_axis(['rank', 'Median'], axis=1), how='left', on=['rank'])
    AggregationTable = AggregationTable.merge(Variable.groupby('rank')['Target'].quantile(q=0.95).reset_index().set_axis(['rank', 'Quantile'], axis=1), how='left', on=['rank'])
    AggregationTable = AggregationTable.merge(Variable.groupby('rank')['Target'].apply(lambda x: len(x)).reset_index().set_axis(['rank', 'NumberOfObserbation'], axis=1), how='left', on=['rank'])

    # Merge to The Dictionary
    Dictionary = Dictionary.merge(AggregationTable, how='left', on=['rank'])

    Dictionary.loc[:, 'Mean'] = Dictionary['Mean'].fillna(np.mean(Variable['Target'].dropna()))
    Dictionary.loc[:, 'Median'] = Dictionary['Median'].fillna(np.median(Variable['Target'].dropna()))
    Dictionary.loc[:, 'Quantile'] = Dictionary['Quantile'].fillna(np.quantile(Variable['Target'].dropna(),q=0.95))
    Dictionary.loc[:, 'NumberOfObserbation'] = Dictionary['NumberOfObserbation'].fillna(0)

    return Dictionary
#Example
# data = pd.read_csv('http://winterolympicsmedals.com/medals.csv')
# VariableToConvert = 'Sport'; TargetName = 'Year';Data = data; FrequencyNumber = 100; Fun = np.median; Suffix='_Median_YMC' 
# Variable = data['Year']
# Target = np.where(data['Year']>=data['Year'].mean(),1,0)
# NumberOfGroups = 10
# print(FourBasicNumericYMC(Variable = data['Year'],Target = np.where(data['Year']>=data['Year'].mean(),1,0),NumberOfGroups = 10))

##  NumericYMC -------------------------------------------------------
def FunNumericYMC(Variable, Target, NumberOfGroups,Fun = np.mean,Name = "Mean"):
    # Create dictionary for Variable
    Dictionary = Ranks_Dictionary(RJitter(Variable,0.00001), ranks_num=NumberOfGroups)
    Dictionary.index = pd.IntervalIndex.from_arrays(Dictionary['lag_value'],
                                                    Dictionary['value'],
                                                    closed='left')
    # Convert Each value in variable to rank
    Variable = pd.DataFrame({'Variable': Variable, 
                             'Target': Target})
    IntervalLocation = Variable['Variable']
    Variable['rank'] = Dictionary.loc[IntervalLocation]['rank'].reset_index(drop=True)
    del IntervalLocation

    # Aggregation Table
    Dictionary = Dictionary.merge(Variable.groupby('rank')['Target'].apply(Fun).reset_index().set_axis(['rank', Name], axis=1), how='left', on=['rank'])

    #Fill NA with the Function outcomes on all the variable
    Dictionary.loc[:, Name] = Dictionary[Name].fillna(Fun(Variable['Target'].dropna()))

    return Dictionary

## Example
# data = pd.read_csv('http://winterolympicsmedals.com/medals.csv')
# VariableToConvert = 'Sport'; TargetName = 'Year';Data = data; FrequencyNumber = 100; Fun = np.median; Suffix='_Median_YMC' 
# Variable = data['Year']
# Target = np.where(data['Year']>=data['Year'].mean(),1,0)
# NumberOfGroups = 10
# NumericYMC(Variable = data['Year'],Target = np.where(data['Year']>=data['Year'].mean(),1,0),NumberOfGroups = 10,Fun = np.median,Name = "Median")

##  Percentile -------------------------------------------------------
def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

#percentile(50)(np.array([1, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5, 4, 5, 5]))

##  Factor YMC -------------------------------------------------------
#data = pd.read_csv('http://winterolympicsmedals.com/medals.csv')
#VariableToConvert = 'Sport'; TargetName = 'Year';Data = data; FrequencyNumber = 100; Fun = np.median; Suffix='_Median_YMC' 
def FunFactorYMC(VariableToConvert, TargetName, Data, FrequencyNumber = 100, Fun = np.mean, Suffix='_Mean_YMC'):

    # Creating variable to transform it to YMC ------------------------
    Variable = Data.loc[:, [TargetName,VariableToConvert]].set_axis(['TargetName','VariableToConvert'], axis=1)
    Variable.loc[:, 'VariableToConvert'] = Variable['VariableToConvert'].astype(str).fillna('NULL')  
   

    # Group all the Not Frequent Factor to one factor group -----------
    NotFrequentFactorGroup = pd.DataFrame(Variable.groupby('VariableToConvert')['TargetName'].apply(lambda x: 'Rare' if len(x) <= FrequencyNumber else 'Frequent')).reset_index()
    NotFrequentFactorGroup.columns = ["VariableName", "SmallGroupOrNot"]
    FrequentFactors = NotFrequentFactorGroup.loc[NotFrequentFactorGroup.SmallGroupOrNot == 'Frequent'].VariableName
    Variable.loc[:, 'VariableToConvert'] = np.where(Variable['VariableToConvert'].isin(FrequentFactors), Variable['VariableToConvert'], 'Not Frequent Factor')

    # Creating Dictionary
    Dictionary_Variable_YMC = Variable.groupby('VariableToConvert')["TargetName"].apply(Fun).reset_index()
    Dictionary_Variable_YMC.columns = ["Variable",TargetName+Suffix]
    Dictionary_Variable_YMC = Dictionary_Variable_YMC.sort_values(by=TargetName+Suffix, ascending=False)

    Dictionary = pd.DataFrame(data = {"VariableToConvert": Data[VariableToConvert].unique()})
    Dictionary['VariableToConvert'] = Dictionary['VariableToConvert'].astype(str).fillna('NULL') 
    Dictionary['Variable'] = np.where(Dictionary['VariableToConvert'].isin(FrequentFactors), Dictionary['VariableToConvert'], 'Not Frequent Factor')
    Dictionary = Dictionary.join(Dictionary_Variable_YMC.set_index('Variable'), how='left', on='Variable')
    Dictionary = Dictionary.drop(columns = 'Variable')
    Dictionary.columns = Dictionary_Variable_YMC.columns

    return Dictionary

#FunFactorYMC(VariableToConvert = 'Sport', TargetName = 'Year',Data = data, FrequencyNumber = 100, Fun = np.median, Suffix='_Median_YMC' )
