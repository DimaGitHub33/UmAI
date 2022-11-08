if __name__ == "__main__":
    print("__main__")
    
    
## Import Packages ---------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from Pridit import preditClassifier
from Functions import Ranks_Dictionary, RJitter, FunFactorYMC, FunNumericYMC

#from warnings import simplefilter
#import random as Random


# Remove the warnings in the console
#simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

## Read Data from local memory ---------------------------------------------------------------------------------
Data = pd.read_parquet('/Users/dhhazanov/Downloads/ppp_v1.parquet.gzip', engine='pyarrow')
# Data = pd.read_parquet(r'C:\github\Utilities\machine_learning_examples\ppp_v1.parquet.gzip', engine='pyarrow')
Data['HAVE_HAKIRA'] = Data['HAVE_HAKIRA'].fillna(-1)



## PRIDIT SCORE ------------------------------------------------------------------------------------------------
preditClassifier = PriditClassifier(Data,conf = {})
#preditClassifier.gen_suprise_order()
priditScore,F,firstEigenVector = preditClassifier.Pridit()
Data['pridit_score'] = priditScore
Data['pridit_score'].describe()

##Rank The Pridit Score
Dictionary = Ranks_Dictionary(RJitter(x = Data['pridit_score'], factor = 0.00001), ranks_num=100)
Dictionary.index = pd.IntervalIndex.from_arrays(Dictionary['lag_value'],Dictionary['value'],closed='left')

# Convert Each value in variable to ranktype(FactorVariables)
Data['Rank'] = Dictionary.loc[Data['pridit_score']]['rank'].reset_index(drop=True)


## Estimation function for mean, median and sum
def aggregations(x):
    Mean = np.mean(x)
    Median = np.median(x)
    Sum = np.sum(x)
    NumberOfObservation = len(x)
    DataReturned = pd.DataFrame({'Mean': [Mean],'Median': [Median],'Sum': [Sum],'NumberOfObservation': [NumberOfObservation]})
    return DataReturned


## Aggregation pridit_score------------------------------------------------------------------------------------------------

AggregationTable_pridit_score = Data.groupby('Rank')['pridit_score'].apply(aggregations).reset_index()
AggregationTable_pridit_score = AggregationTable_pridit_score.drop(columns=['level_1'])
print(AggregationTable_pridit_score)


## Aggregation HAVE_HAKIRA
AggregationTable_HAVE_HAKIRA = Data.groupby('Rank')['HAVE_HAKIRA'].apply(aggregations).reset_index()
AggregationTable_HAVE_HAKIRA = AggregationTable_HAVE_HAKIRA.drop(columns=['level_1'])
print(AggregationTable_HAVE_HAKIRA)

AggregationTable_HAVE_TVIA = Data.groupby('Rank')['HAVE_TVIA'].apply(aggregations).reset_index()
AggregationTable_HAVE_TVIA = AggregationTable_HAVE_TVIA.drop(columns=['level_1'])
print(AggregationTable_HAVE_TVIA)

## FunFactorYMC------------------------------------------------------------------------------------------------
#print(np.where(Data['pridit_score'] >= Data['pridit_score'].mean(),1,0))
Data['Target'] = np.where(Data['pridit_score'] >= Data['pridit_score'].mean(),1,0)
print(Data['Target'])

Data.re
Data.to_csv('Out.csv',index=False)
#print(FunFactorYMC(VariableToConvert = 'HAVE_HAKIRA', TargetName = 'Target',Data = Data, FrequencyNumber = 100, Fun = np.median, Suffix='_Median_YMC' ))


## Model -------------------------------------------------------------------------------------------------------
Data['Target'] = np.where(Data['pridit_score'] >= Data['pridit_score'].mean(),1,0)
from Model import Model
conf={
    'Path':'/Users/dhhazanov/UmAI/Models/Model.pckl'
}
Model(Data,conf)

## Model -------------------------------------------------------------------------------------------------------
from sklearn.datasets import load_breast_cancer
from Pridit import PriditClassifier
breast_cancer_x,breast_cancer_y = load_breast_cancer(return_X_y=True)



# conf = {
#     'UsingFacotr': 'OnlyVariables',  ##Both, OnlyVariables, None
#     'FactorVariables': FactorVariables,  ##List, None
#     'NumericVariables': NumericVariables,  ##list, None
#     #'FactorVariables': [],  ##List, None
#     #'NumericVariables': [],  ##list, None
#     'FactorsVariablesOrder': None,  ##List, None
#     'NumericVariablesOrder': None  ##List, None
# }

PC = PriditClassifier(pd.DataFrame(breast_cancer_x), conf={})
#preditClassifier = PreditClassifier(Data, conf=conf)
priditScore,F,firstEigenVector  = PC.Pridit()

Data = pd.concat([pd.DataFrame(breast_cancer_x),pd.DataFrame({'Y':breast_cancer_y})],axis=1)
Data['priditScore'] = priditScore
Data['priditScore'].describe()


##Rank The Pridit Score
Dictionary = Ranks_Dictionary(RJitter(x = Data['priditScore'], factor = 0.00001), ranks_num=10)
Dictionary.index = pd.IntervalIndex.from_arrays(Dictionary['lag_value'],Dictionary['value'],closed='left')

# Convert Each value in variable to ranktype(FactorVariables)
Data['Rank'] = Dictionary.loc[Data['priditScore']]['rank'].reset_index(drop=True)
Data['Rank'] = np.where(Data['priditScore']<=0,-1,1)


## Estimation function for mean, median and sum
def aggregations(x):
    Mean = np.mean(x)
    Median = np.median(x)
    Sum = np.sum(x)
    NumberOfObservation = len(x)
    DataReturned = pd.DataFrame({'Mean': [Mean],'Median': [Median],'Sum': [Sum],'NumberOfObservation': [NumberOfObservation]})
    return DataReturned

## Aggregation HAVE_HAKIRA
AggregationTable_priditScore = Data.groupby('Rank')['priditScore'].apply(aggregations).reset_index()
AggregationTable_priditScore = AggregationTable_priditScore.drop(columns=['level_1'])
print(AggregationTable_priditScore)

AggregationTable_Y = Data.groupby('Rank')['Y'].apply(aggregations).reset_index()
AggregationTable_Y= AggregationTable_Y.drop(columns=['level_1'])
print(AggregationTable_Y)

100*np.mean(breast_cancer_y)






from sklearn.datasets import make_classification
make_classification_x,make_classification_y = make_classification(n_samples=10000)



PC = PriditClassifier(pd.DataFrame(make_classification_x), conf={})
#preditClassifier = PreditClassifier(Data, conf=conf)
priditScore,F,firstEigenVector  = PC.Pridit()

Data = pd.concat([pd.DataFrame(make_classification_x),pd.DataFrame({'Y':make_classification_y})],axis=1)
Data['priditScore'] = priditScore
Data['priditScore'].describe()


##Rank The Pridit Score
Dictionary = Ranks_Dictionary(RJitter(x = Data['priditScore'], factor = 0.00001), ranks_num=10)
Dictionary.index = pd.IntervalIndex.from_arrays(Dictionary['lag_value'],Dictionary['value'],closed='left')

# Convert Each value in variable to ranktype(FactorVariables)
Data['Rank'] = Dictionary.loc[Data['priditScore']]['rank'].reset_index(drop=True)
Data['Rank'] = np.where(Data['priditScore']<=0,-1,1)

## Estimation function for mean, median and sum
def aggregations(x):
    Mean = np.mean(x)
    Median = np.median(x)
    Sum = np.sum(x)
    NumberOfObservation = len(x)
    DataReturned = pd.DataFrame({'Mean': [Mean],'Median': [Median],'Sum': [Sum],'NumberOfObservation': [NumberOfObservation]})
    return DataReturned


AggregationTable_Y = Data.groupby('Rank')['Y'].apply(aggregations).reset_index()
AggregationTable_Y= AggregationTable_Y.drop(columns=['level_1'])
print(AggregationTable_Y)

100*np.mean(make_classification_y)