import pandas as pd
import numpy as np
import random as Random
from warnings import simplefilter

## Remove the warnings in the console --------------------------------------------
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

class PreditClassifier():
    def __init__(self, Data, conf):
        self.Data = Data
        self.conf = conf

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

    def Ranks_Dictionary(self, temp_data, ranks_num):
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

    def RJitter(self, x, factor):
        z = max(x) - min(x)
        amount = factor * (z / 50)
        x = x + np.random.uniform(-amount, amount, len(x))
        return (x)

    ## Pridit ----------------------------------------------------------------------
    """    
    Pridit
        Args:
          Data:
            Data frame of numerical and factorial data
            Example:
                                    ID   DATE_OF_BIRTH GENDER 
                              14262240      ז      1946-11-15
                              14262455      ז      1956-04-18
                              14263677      ז      1953-03-15
                              14263727      נ      1958-02-12
                              14265052      נ      1956-04-24
    
          FactorVariables:
            List of all the variables that their type is factorial
            Example:
            ['GENDER', 'FAMILY_STATUS']
          NumericVariables:
            List of all the variables that their type is numerical
            Example:
            ['Number_Of_Kids', 'Age']
          FactorsVariablesOrder:
            data frame of all the factor variables and their levels order
            Example:
                     Variable               Level  Order
                       GENDER                   ז      0
                       GENDER                   נ      1
                FAMILY_STATUS                   נ      0
                FAMILY_STATUS                   ר      1
                FAMILY_STATUS                   א      2
          NumericVariablesOrder
            data frame of all the numeric variables and their sign order
            Example:
                    Variable  Order
                         Age      1
                      Salery      1
                      Height      0
                      weight      1
    
        Returns:
          Pridit Score
          Example:
            Data = pd.read_parquet('/Downloads/ppp.parquet.gzip', engine='pyarrow')
            PriditScore = Pridit(Data)
            print(PriditScore)
            [-0.63490772, -0.15769004, -0.54438071, ..., -0.60417859,-0.42238741,  9.05145987]
    
    """

    def Pridit(self):
        
        Data = self.Data
        
        ## Fill Configuration -----------------------------------------------------
        if (not 'UsingFacotr' in self.conf):
            self.conf['UsingFacotr'] = None
        if (not 'FactorVariables' in self.conf or self.conf['FactorVariables'] == None):
            self.conf['FactorVariables'] = []
            factor_variables = self.conf['FactorVariables']
        if (not 'NumericVariables' in self.conf or self.conf['NumericVariables'] == None):
            self.conf['NumericVariables'] = []
            numeric_variables = self.conf['NumericVariables']
        if (not 'FactorsVariablesOrder' in self.conf):
            self.conf['FactorsVariablesOrder'] = None
        if (not 'NumericVariablesOrder' in self.conf):
            self.conf['NumericVariablesOrder'] = None
        if (not 'UsingFacotr' in self.conf):
            self.conf['NumericVariablesOrder'] = None

        if (self.conf['UsingFacotr'] == 'OnlyVariables'):
            factor_variables = self.conf['FactorVariables']
            numeric_variables = self.conf['NumericVariables']

        ## Fill the FactorVariables and NumericVariables list for other columns in the input data ----
        if (self.conf['UsingFacotr'] == 'Both'):

            factor_variables = self.conf['FactorVariables']
            numeric_variables = self.conf['NumericVariables']

            factor_variables2 = []
            data_types = Data.dtypes.reset_index().rename(columns={'index': 'Index', 0: 'Type'})
            for Index, row in data_types.iterrows():
                if row['Type'] in ['object', 'str']:
                    factor_variables2.append(row['Index'])

            factor_variables2 = [i for i in factor_variables2 if i not in numeric_variables]
            factor_variables2 = [i for i in factor_variables2 if i not in factor_variables]
            if (len(factor_variables2) > 0):
                factor_variables.extend(factor_variables2)

            numeric_variables2 = []
            data_types = Data.dtypes.reset_index().rename(columns={'index': 'Index', 0: 'Type'})
            for Index, row in data_types.iterrows():
                if row['Type'] in ['int64', 'float64']:
                    numeric_variables2.append(row['Index'])

            numeric_variables2 = [i for i in numeric_variables2 if i not in numeric_variables]
            numeric_variables2 = [i for i in numeric_variables2 if i not in factor_variables]
            if (len(numeric_variables2) > 0):
                numeric_variables.extend(numeric_variables2)

            del (numeric_variables2)
            del (factor_variables2)

        ## Fill the FactorVariables and NumericVariables list ----------------------
        if factor_variables is None or len(factor_variables)==0:
            factor_variables = []
            data_types = Data.dtypes.reset_index().rename(columns={'index': 'Index', 0: 'Type'})
            for Index, row in data_types.iterrows():
                if row['Type'] in ['object', 'str']:
                    factor_variables.append(row['Index'])

        if numeric_variables is None or len(numeric_variables)==0:
            numeric_variables = []
            data_types = Data.dtypes.reset_index().rename(columns={'index': 'Index', 0: 'Type'})
            for Index, row in data_types.iterrows():
                if row['Type'] in ['int64', 'float64']:
                    numeric_variables.append(row['Index'])

        ## Fill the orders of the variables
        factors_variables_order = self.conf['FactorsVariablesOrder']
        numeric_variables_order = self.conf['NumericVariablesOrder']

        ## F calculation for Factor variables  ------------------------------------
        F = pd.DataFrame()
        for variable_to_convert in factor_variables:
            # print(VariableToConvert)
            variable = Data[[variable_to_convert]].copy()
            variable.columns = ["VariableToConvert"]
            variable.loc[:, 'VariableToConvert'] = variable['VariableToConvert'].astype(str).fillna('NULL')

            # Frequency table
            if (len(variable['VariableToConvert'].unique()) < 2):
                continue

            frequency_table = pd.DataFrame(variable['VariableToConvert'].value_counts(normalize=True)).reset_index()
            frequency_table.columns = [variable_to_convert, 'Frequency']

            ## Order the Factors by the FactorsVariablesOrder
            if factors_variables_order is None:
                frequency_table = frequency_table.sort_values('Frequency', ascending=True)
            else:
                Order = factors_variables_order[factors_variables_order['Variable'] == variable_to_convert].set_index('Level')
                if len(Order) == 0:
                    frequency_table = frequency_table.sort_values('Frequency', ascending=True)
                else:
                    frequency_table = frequency_table.join(Order, on=variable_to_convert, how='left')
                    frequency_table['Order'] = frequency_table['Order'].fillna(np.mean(frequency_table['Order']))
                    frequency_table = frequency_table.sort_values('Order', ascending=True)

            ##Calculating the weights after ordering the Levels
            frequency_table['CumSum'] = frequency_table['Frequency'].cumsum()
            frequency_table['F'] = frequency_table['CumSum'] - frequency_table['Frequency'] - (1 - frequency_table['CumSum'])
            frequency_table = frequency_table[[variable_to_convert, 'F']]
            frequency_table.columns = [variable_to_convert, 'FTransformation_' + variable_to_convert]

            # Merge to The Table
            F[variable_to_convert] = Data[variable_to_convert].astype(str)
            F = F.join(frequency_table.set_index(variable_to_convert), on=variable_to_convert, how='left')
            F = F.drop(variable_to_convert, axis=1)

        ## F calculation for numeric variables ------------------------------------
        for variable_to_convert in [NV for NV in numeric_variables if NV not in factor_variables]:
            # print(VariableToConvert)
            variable = Data[[variable_to_convert]].copy().astype(float)
            variable = variable.fillna(np.mean(variable, axis=0))
            variable.columns = ["VariableToConvert"]

            # Rank the numeric variable
            dictionary = self.Ranks_Dictionary(self.RJitter(variable['VariableToConvert'], 0.00001), ranks_num=10)
            dictionary.index = pd.IntervalIndex.from_arrays(dictionary['lag_value'],
                                                            dictionary['value'],
                                                            closed='left')

            # Convert Each value in variable to rank
            variable['Rank'] = dictionary.loc[variable['VariableToConvert']]['rank'].reset_index(drop=True).astype(str)

            # Frequency table
            if (len(variable['VariableToConvert'].unique()) < 2):
                continue

            frequency_table = pd.DataFrame(variable['Rank'].value_counts(normalize=True)).reset_index()
            frequency_table.columns = ['Rank', 'Frequency']
            frequency_table['Rank'] = frequency_table['Rank'].astype(float)

            ## Order the Factors by the NumericVariablesOrder
            if factors_variables_order is None:
                frequency_table = frequency_table.sort_values('Frequency', ascending=True)
            else:
                Order = numeric_variables_order[numeric_variables_order['Variable'] == variable_to_convert]
                if len(Order) == 0:
                    frequency_table = frequency_table.sort_values('Frequency', ascending=True)
                else:
                    if Order['Order'][0] == 0:
                        frequency_table = frequency_table.sort_values('Rank', ascending=False)
                    else:
                        frequency_table = frequency_table.sort_values('Rank', ascending=True)

            ##Calculating the weights after ordering the numeric levels
            frequency_table['CumSum'] = frequency_table['Frequency'].cumsum().copy()
            frequency_table['F'] = frequency_table['CumSum'] - frequency_table['Frequency'] - (1 - frequency_table['CumSum'])
            frequency_table = frequency_table[['Rank', 'F']]
            frequency_table.columns = ['Rank', 'FTransformation_' + variable_to_convert]
            frequency_table['Rank'] = frequency_table['Rank'].astype(int).astype(str)

            # Merge to The Table
            variable = variable.join(frequency_table.set_index('Rank'), on='Rank', how='left')
            F['FTransformation_' + variable_to_convert] = variable['FTransformation_' + variable_to_convert]

        ## Calculating the Eigenvector of the maximum eigenvalues-------------------
        F_mat = F.to_numpy()
        F_t_F = np.matmul(F_mat.T, F_mat)
        eigenvalues, eigenvectors = np.linalg.eigh(F_t_F)
        pridit_score = F_mat.dot(eigenvectors[:, np.argmax(eigenvalues)])

        return pridit_score

    def gen_rank(self, Data):
        Dictionary = self.Ranks_Dictionary(self.RJitter(Data['pridit_score'], 0.00001), ranks_num=100)
        Dictionary.index = pd.IntervalIndex.from_arrays(Dictionary['lag_value'],
                                                        Dictionary['value'],
                                                        closed='left')

        # Convert Each value in variable to ranktype(FactorVariables)

        self.DataWithRank = self.Data
        self.DataWithRank['Rank'] = Dictionary.loc[Data['pridit_score']]['rank'].reset_index(drop=True)

        return self.DataWithRank

    def aggregations(self, x):
        Mean = np.mean(x)
        Median = np.median(x)
        Sum = np.sum(x)
        NumberOfObservation = len(x)
        DataReturned = pd.DataFrame({'Mean': [Mean],
                                     'Median': [Median],
                                     'Sum': [Sum],
                                     'NumberOfObservation': [NumberOfObservation]})
        return DataReturned

    def check_score_to_column(self, column, aggregation=None):

        aggregation_table_pridit_score = self.DataWithRank.groupby('Rank')[column].apply(self.aggregations).reset_index()
        aggregation_table_pridit_score = aggregation_table_pridit_score.drop(columns=['level_1'])
        return aggregation_table_pridit_score

    def gen_suprise_order(self):
        ## Creating FactorsVariablesOrder for each factor variable it will randomized the order of the levels

        if (not self.conf['FactorVariables'] == None and len(self.conf['FactorVariables'])>0):
            self.conf['FactorsVariablesOrder'] = pd.DataFrame()
            for variable_name in self.conf['FactorVariables']:
                rows = pd.DataFrame({'Variable': variable_name,
                                     'Level': Data[variable_name].unique(),
                                     'Order': [number for number in range(0, len(Data[variable_name].unique()))]})
                self.conf['FactorsVariablesOrder'] = pd.concat([self.conf['FactorsVariablesOrder'], rows])
        else:
            self.conf['FactorsVariablesOrder'] = None

        ## Creating NumericVariablesOrder for each numeric variable it will be randomized the sign of the variable
        if (not self.conf['NumericVariables'] == None  and len(self.conf['NumericVariables'])>0):
            self.conf['NumericVariablesOrder'] = pd.DataFrame()

            for variable in self.conf['NumericVariables']:
                rows = pd.DataFrame({'Variable': variable,
                                     'Order': Random.randint(0, 1)}, index=[0])
                self.conf['NumericVariablesOrder'] = pd.concat([self.conf['NumericVariablesOrder'], rows])
        else:
            self.conf['NumericVariablesOrder'] = None
        return self.conf


## -----------------------------------------------------------------------------
## -------------------------- Run Pridit Score function ------------------------
## -----------------------------------------------------------------------------

# # Import libraries
# import pyarrow.parquet as pq
# from warnings import simplefilter
# import random as Random

# # Remove the warnings in the console
# simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# ## Read Data from my local memory
# # Data = pd.read_parquet('/Users/dhhazanov/Downloads/ppp.parquet.gzip', engine='pyarrow')
# Data = pd.read_parquet('/Users/dhhazanov/Downloads/ppp_v1.parquet.gzip', engine='pyarrow')
# #Data = pd.read_parquet(r'C:\github\Utilities\machine_learning_examples\ppp_v1.parquet.gzip', engine='pyarrow')
# Data['HAVE_HAKIRA'] = Data['HAVE_HAKIRA'].fillna(-1)

# ## Run the pridit Score without extra argument like FactorVariables,NumericVariables,FactorsVariablesOrder,NumericVariablesOrder

# FactorVariables = ['GENDER', 'FAMILY_STATUS', 'ACADEMIC_DEGREE', 'PROFESSION', 'TEUR_ISUK', 'ISUK_MERAKEZ', 'TEUR_TACHBIV',
#                    'ADDRESS', 'STREET', 'CITY', 'TEUR_EZOR', 'MIKUD_BR', 'YESHUV_BR', 'TEUR_EZOR_MIKUD', 'TEUR_TAT_EZOR_MIKUD',
#                    'GEOCODE_TYPE', 'PHONES', 'CELLULARS', 'ASIRON_LAMAS', 'M_SOCHEN_MOCHER', 'SHEM_SOCHNUT_MOCHER', 'M_ERUAS']
# NumericVariables = ['TEOUDAT_ZEOUT', 'GIL', 'BR_FLG_YELED', 'CHILD_COUNT', 'ISUK', 'STATUS_ISUK', 'ZAVARON', 'TACHBIV', 'ISHUN',
#                     'SUG_VIP', 'CITY_ID', 'KOD_EZOR', 'ZIP_CODE', 'GEOCODEX', 'GEOCODEY', 'ESHKOL_PEREFIRIA', 'ESHKOL_LAMAS', 'REPORTEDSALARY',
#                     'VETEK', 'VETEK_PAIL', 'BR_FLG_POLISAT_KOLEKTIV', 'HAVE_BRIUT', 'BR_KAMUT_MUTZRIM_PEILIM', 'BR_FLG_CHOV', 'BR_SCHUM_CHOV']

# conf = {
#     'UsingFacotr': 'OnlyVariables',  ##Both, OnlyVariables, None
#     'FactorVariables': FactorVariables,  ##List, None
#     'NumericVariables': NumericVariables,  ##list, None
#     #'FactorVariables': [],  ##List, None
#     #'NumericVariables': [],  ##list, None
#     'FactorsVariablesOrder': None,  ##List, None
#     'NumericVariablesOrder': None  ##List, None
# }

# preditClassifier = PreditClassifier(Data, conf={})
# preditClassifier.gen_suprise_order()
# pridit_score = preditClassifier.Pridit()
# Data['pridit_score'] = pridit_score
# Data['pridit_score'].describe()
# print(pridit_score)

# DataWithRank = preditClassifier.gen_rank(Data)
# print(preditClassifier.check_score_to_column('HAVE_HAKIRA'))
# print(preditClassifier.check_score_to_column('HAVE_TVIA')) 

# print(preditClassifier.check_score_to_column('HAVE_HAKIRA'))
## Run the pridit Score without With extra argument like FactorVariables,NumericVariables,FactorsVariablesOrder,NumericVariablesOrder

## FactorVariables and NumericVariables list

#
# pridit_score = Pridit(Data, conf)
# Data['pridit_score'] = pridit_score
# Data['pridit_score'].describe()
# print(pridit_score)
#
# ## Creating FactorsVariablesOrder for each factor variable it will randomized the order of the levels
# FactorsVariablesOrder = pd.DataFrame()
# for VariableName in FactorVariables:
#     Rows = pd.DataFrame({'Variable': VariableName,
#                          'Level': Data[VariableName].unique(),
#                          'Order': [Number for Number in range(0, len(Data[VariableName].unique()))]})
#     FactorsVariablesOrder = pd.concat([FactorsVariablesOrder, Rows])
#
# ## Creating NumericVariablesOrder for each numeric variable it will be randomized the sign of the variable
# NumericVariablesOrder = pd.DataFrame()
# for Variable in NumericVariables:
#     Rows = pd.DataFrame({'Variable': Variable,
#                          'Order': Random.randint(0, 1)}, index=[0])
#     NumericVariablesOrder = pd.concat([NumericVariablesOrder, Rows])
#
# ## Run the pridit score with the extra argument
# pridit_score = Pridit(Data, FactorVariables, NumericVariables, FactorsVariablesOrder, NumericVariablesOrder)
#
# ## -----------------------------------------------------------------------------
# ## -------------------------- Check the pridit score ---------------------------
# ## -----------------------------------------------------------------------------
#
# ##Rank The Pridit Score
# Dictionary = Ranks_Dictionary(RJitter(Data['pridit_score'], 0.00001), ranks_num=100)
# Dictionary.index = pd.IntervalIndex.from_arrays(Dictionary['lag_value'],
#                                                 Dictionary['value'],
#                                                 closed='left')
#
# # Convert Each value in variable to ranktype(FactorVariables)
# Data['Rank'] = Dictionary.loc[Data['pridit_score']]['rank'].reset_index(drop=True)
#
#
# ## Estimation function for mean, median and sum
# def aggregations(x):
#     Mean = np.mean(x)
#     Median = np.median(x)
#     Sum = np.sum(x)
#     NumberOfObservation = len(x)
#     DataReturned = pd.DataFrame({'Mean': [Mean],
#                                  'Median': [Median],
#                                  'Sum': [Sum],
#                                  'NumberOfObservation': [NumberOfObservation]})
#     return DataReturned
#
#
# # Aggregation Suspicious
# AggregationTable_pridit_score = Data.groupby('Rank')['pridit_score'].apply(aggregations).reset_index()
# AggregationTable_pridit_score = AggregationTable_pridit_score.drop(columns=['level_1'])
#
# # Aggregation Suspicious
# AggregationTable_HAVE_HAKIRA = Data.groupby('Rank')['HAVE_HAKIRA'].apply(aggregations).reset_index()
# AggregationTable_HAVE_HAKIRA = AggregationTable_HAVE_HAKIRA.drop(columns=['level_1'])
#
# # Aggregation Suspicious_Money
# AggregationTable_Suspicious_HAVE_TVIA = Data.groupby('Rank')['HAVE_TVIA'].apply(aggregations).reset_index()
# AggregationTable_Suspicious_HAVE_TVIA = AggregationTable_Suspicious_HAVE_TVIA.drop(columns=['level_1'])
