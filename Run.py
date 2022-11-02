if __name__ == "__main__":
    print("__main__")
    
    
## Import Packages ---------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from PriditClass import PreditClassifier
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
preditClassifier = PreditClassifier(Data,conf = {})
#preditClassifier.gen_suprise_order()
pridit_score = preditClassifier.Pridit()
Data['pridit_score'] = pridit_score
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

