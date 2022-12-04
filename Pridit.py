import pandas as pd
import numpy as np
import random as Random
from warnings import simplefilter
from sklearn.decomposition import PCA
from Functions import Ranks_Dictionary, RJitter, FunFactorYMC, FunNumericYMC
import logging as logger

## Remove the warnings in the console --------------------------------------------
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


class PriditClassifier():
    def __init__(self, Data, conf, logger):

        self.Data = Data
        self.conf = conf
        self.logger = logger

        self.logger.debug('PriditClassifier was created')

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
          Pridit Score, F, EigenVector
          Example:
            Data = pd.read_parquet('/Downloads/ppp.parquet.gzip', engine='pyarrow')
            PriditScore = Pridit(Data)
            print(PriditScore)
            [-0.63490772, -0.15769004, -0.54438071, ..., -0.60417859,-0.42238741,  9.05145987]
    
    """

    def Pridit(self):
        conf = self.conf
        Data = self.Data
        logger = self.logger
        logger.debug('pridit called with parameters conf={conf} '.format(conf = conf))

        logger.info('check conf params ')
        ## Fill Configuration -----------------------------------------------------
        if (not 'UsingFactor' in conf):
            logger.info('UsingFactor not found in conf using default -> None')
            conf['UsingFactor'] = None
        if (not 'FactorVariables' in conf or conf['FactorVariables'] == None):
            logger.info('FactorVariables not found in conf using default -> []')
            conf['FactorVariables'] = []
            factorVariables = conf['FactorVariables']
        if (not 'NumericVariables' in conf or conf['NumericVariables'] == None):
            logger.info('NumericVariables not found in conf using default -> []')
            conf['NumericVariables'] = []
            numericVariables = conf['NumericVariables']
        if (not 'FactorsVariablesOrder' in conf):
            logger.info('FactorsVariablesOrder not found in conf using default -> None')
            conf['FactorsVariablesOrder'] = None
        if (not 'NumericVariablesOrder' in conf):
            logger.info('NumericVariablesOrder not found in conf using default -> None')
            conf['NumericVariablesOrder'] = None
        if (not 'UsingFactor' in conf):
            logger.info('UsingFactor not found in conf using default -> None]')
            conf['NumericVariablesOrder'] = None

        if (conf['UsingFactor'] == 'OnlyVariables'):
            factorVariables = conf['FactorVariables']
            numericVariables = conf['NumericVariables']
            
        ## Remove all the unique columns in Data
        logger.info("Removing all the unique columns in the data")
        for colName in Data.columns:
            if len(Data.loc[:,colName].unique())==1:
                logger.info(colName + " Removed")
                Data.drop(colName, inplace=True, axis=1)
        
        logger.info('execute pridit with conf -> {conf}'.format(conf=conf))

        
        ## Fill the FactorVariables and NumericVariables list for other columns in the input data ----
        if (conf['UsingFactor'] == 'Both'):
            logger.info('UsingFactor == Both using factors [FactorVariables + NumericVariables]'.format(conf=conf))

            factorVariables = conf['FactorVariables']
            numericVariables = conf['NumericVariables']

            factorVariables2 = []
            dataTypes = Data.dtypes.reset_index().rename(columns={'index': 'Index', 0: 'Type'})
            for Index, row in dataTypes.iterrows():
                if row['Type'] in ['object', 'str']:
                    factorVariables2.append(row['Index'])

            factorVariables2 = [i for i in factorVariables2 if i not in numericVariables]
            factorVariables2 = [i for i in factorVariables2 if i not in factorVariables]
            if (len(factorVariables2) > 0):
                factorVariables.extend(factorVariables2)

            numericVariables2 = []
            dataTypes = Data.dtypes.reset_index().rename(columns={'index': 'Index', 0: 'Type'})
            for Index, row in dataTypes.iterrows():
                if row['Type'] in ['int64', 'float64']:
                    numericVariables2.append(row['Index'])

            numericVariables2 = [i for i in numericVariables2 if i not in numericVariables]
            numericVariables2 = [i for i in numericVariables2 if i not in factorVariables]
            if (len(numericVariables2) > 0):
                numericVariables.extend(numericVariables2)

            del (numericVariables2)
            del (factorVariables2)

        ## Fill the FactorVariables and NumericVariables list ----------------------
        if factorVariables is None or len(factorVariables) == 0:
            logger.info('factorVariables is None -> build auto factor variable'.format(conf=conf))
            factorVariables = []
            dataTypes = Data.dtypes.reset_index().rename(columns={'index': 'Index', 0: 'Type'})
            for Index, row in dataTypes.iterrows():
                if row['Type'] in ['object', 'str']:
                    factorVariables.append(row['Index'])

        if numericVariables is None or len(numericVariables) == 0:
            logger.info('numericVariables is None -> build auto factor variable'.format(conf=conf))
            numericVariables = []
            dataTypes = Data.dtypes.reset_index().rename(columns={'index': 'Index', 0: 'Type'})
            for Index, row in dataTypes.iterrows():
                if row['Type'] in ['int64', 'float64']:
                    numericVariables.append(row['Index'])

        ## Fill the orders of the variables
        factorsVariablesOrder = conf['FactorsVariablesOrder']
        logger.info('using factorsVariablesOrder with {conf}'.format(conf=factorsVariablesOrder))
        numericVariablesOrder = conf['NumericVariablesOrder']
        logger.info('using NumericVariablesOrder with {conf}'.format(conf=numericVariablesOrder))

        ## F calculation for Factor variables  ------------------------------------
        F = pd.DataFrame()
        for variableToConvert in factorVariables:
            # print(VariableToConvert)
            variable = Data[[variableToConvert]].copy()
            variable.columns = ["VariableToConvert"]
            variable.loc[:, 'VariableToConvert'] = variable['VariableToConvert'].astype(str).fillna('NULL')

            # Frequency table
            if (len(variable['VariableToConvert'].unique()) < 2):
                continue

            frequencyTable = pd.DataFrame(variable['VariableToConvert'].value_counts(normalize=True)).reset_index()
            frequencyTable.columns = [variableToConvert, 'Frequency']

            ## Order the Factors by the FactorsVariablesOrder
            if factorsVariablesOrder is None:
                frequencyTable = frequencyTable.sort_values('Frequency', ascending=True)
            else:
                Order = factorsVariablesOrder[factorsVariablesOrder['Variable'] == variableToConvert].set_index('Level')
                if len(Order) == 0:
                    frequencyTable = frequencyTable.sort_values('Frequency', ascending=True)
                else:
                    frequencyTable = frequencyTable.join(Order, on=variableToConvert, how='left')
                    frequencyTable['Order'] = frequencyTable['Order'].fillna(np.mean(frequencyTable['Order']))
                    frequencyTable = frequencyTable.sort_values('Order', ascending=True)

            ##Calculating the weights after ordering the Levels
            frequencyTable['CumSum'] = frequencyTable['Frequency'].cumsum()
            frequencyTable['F'] = frequencyTable['CumSum'] - frequencyTable['Frequency'] - (1 - frequencyTable['CumSum'])
            frequencyTable = frequencyTable[[variableToConvert, 'F']]
            frequencyTable.columns = [variableToConvert, 'FTransformation_' + variableToConvert]

            # Merge to The Table
            F[variableToConvert] = Data[variableToConvert].astype(str)
            F = F.join(frequencyTable.set_index(variableToConvert), on=variableToConvert, how='left')
            F = F.drop(variableToConvert, axis=1)

        ## F calculation for numeric variables ------------------------------------
        for variableToConvert in [NV for NV in numericVariables if NV not in factorVariables]:
            # print(VariableToConvert)
            variable = Data[[variableToConvert]].copy().astype(float)
            variable = variable.fillna(np.mean(variable, axis=0))
            variable.columns = ["VariableToConvert"]

            # Rank the numeric variable
            dictionary = Ranks_Dictionary(RJitter(variable['VariableToConvert'], 0.00001), ranks_num=10)
            dictionary.index = pd.IntervalIndex.from_arrays(dictionary['lag_value'],
                                                            dictionary['value'],
                                                            closed='left')

            # Convert Each value in variable to rank
            variable['Rank'] = dictionary.loc[variable['VariableToConvert']]['rank'].reset_index(drop=True).astype(str)

            # Frequency table
            if (len(variable['VariableToConvert'].unique()) < 2):
                continue

            frequencyTable = pd.DataFrame(variable['Rank'].value_counts(normalize=True)).reset_index()
            frequencyTable.columns = ['Rank', 'Frequency']
            frequencyTable['Rank'] = frequencyTable['Rank'].astype(float)

            ## Order the Factors by the NumericVariablesOrder
            if numericVariablesOrder is None:
                frequencyTable = frequencyTable.sort_values('Rank', ascending=True)
            else:
                Order = numericVariablesOrder[numericVariablesOrder['Variable'] == variableToConvert]
                if len(Order) == 0:
                    frequencyTable = frequencyTable.sort_values('Rank', ascending=True)
                else:
                    if Order['Order'][0] == 0:
                        frequencyTable = frequencyTable.sort_values('Rank', ascending=False)
                    else:
                        frequencyTable = frequencyTable.sort_values('Rank', ascending=True)

            ##Calculating the weights after ordering the numeric levels
            frequencyTable['CumSum'] = frequencyTable['Frequency'].cumsum().copy()
            frequencyTable['F'] = frequencyTable['CumSum'] - frequencyTable['Frequency'] - (1 - frequencyTable['CumSum'])
            frequencyTable = frequencyTable[['Rank', 'F']]
            frequencyTable.columns = ['Rank', 'FTransformation_' + str(variableToConvert)]
            frequencyTable['Rank'] = frequencyTable['Rank'].astype(int).astype(str)

            # Merge to The Table
            variable = variable.join(frequencyTable.set_index('Rank'), on='Rank', how='left')
            F['FTransformation_' + str(variableToConvert)] = variable['FTransformation_' + str(variableToConvert)]

        ## Calculating the Eigenvector of the maximum eigenvalues-------------------
        FMat = F.to_numpy()
        FTF = np.matmul(FMat.T, FMat)  ##This is the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(FTF)
        firstEigenVector = eigenvectors[:, np.argmax(eigenvalues)]
        priditScore = FMat.dot(firstEigenVector)
        
        ## Calculating the Eigenvector of the maximum eigenvalues (With PCA)--------
        # Apply PCA
        #pca = PCA(n_components=1)
        #X = pca.fit_transform(F)

        #loadings = pd.DataFrame(pca.components_.T, columns=['PC1'], index=F.columns)
        #PC1 = loadings.PC1
        #priditScore = FMat.dot(PC1)

        return priditScore,F,firstEigenVector

# -----------------------------------------------------------------------------
# -------------------------- Run Pridit Score function ------------------------
# -----------------------------------------------------------------------------
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

# # ## Creating FactorsVariablesOrder for each factor variable it will randomized the order of the levels
# # FactorsVariablesOrder = pd.DataFrame()
# # for VariableName in FactorVariables:
# #     Rows = pd.DataFrame({'Variable': VariableName,
# #                          'Level': Data[VariableName].unique(),
# #                          'Order': [Number for Number in range(0, len(Data[VariableName].unique()))]})
# #     FactorsVariablesOrder = pd.concat([FactorsVariablesOrder, Rows])

# # ## Creating NumericVariablesOrder for each numeric variable it will be randomized the sign of the variable
# # NumericVariablesOrder = pd.DataFrame()
# # for Variable in NumericVariables:
# #     Rows = pd.DataFrame({'Variable': Variable,
# #                          'Order': Random.randint(0, 1)}, index=[0])
# #     NumericVariablesOrder = pd.concat([NumericVariablesOrder, Rows])

# conf = {
#     'UsingFactor': 'OnlyVariables',  ##Both, OnlyVariables, None
#     'FactorVariables': FactorVariables,  ##List, None
#     'NumericVariables': NumericVariables,  ##list, None
#     #'FactorVariables': [],  ##List, None
#     #'NumericVariables': [],  ##list, None
#     'FactorsVariablesOrder': None,  ##List, None
#     'NumericVariablesOrder': None  ##List, None
# }

# PriditClassifier = PriditClassifier(Data, conf={})
# #PriditClassifier = PriditClassifier(Data, conf=conf)
# priditScore,F,firstEigenVector  = PriditClassifier.Pridit()
# Data['priditScore'] = priditScore
# Data['priditScore'].describe()
# print(priditScore)


# ## -----------------------------------------------------------------------------
# ## -------------------------- Check the pridit score ---------------------------
# ## -----------------------------------------------------------------------------

# ##Rank The Pridit Score
# Dictionary = Ranks_Dictionary(RJitter(Data['priditScore'], 0.00001), ranks_num=10)
# Dictionary.index = pd.IntervalIndex.from_arrays(Dictionary['lag_value'],
#                                                 Dictionary['value'],
#                                                 closed='left')

# # Convert Each value in variable to ranktype(FactorVariables)
# Data['Rank'] = Dictionary.loc[Data['priditScore']]['rank'].reset_index(drop=True)


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


# # Aggregation Suspicious
# AggregationTable_priditScore = Data.groupby('Rank')['priditScore'].apply(aggregations).reset_index()
# AggregationTable_priditScore = AggregationTable_priditScore.drop(columns=['level_1'])

# # Aggregation Suspicious
# AggregationTable_HAVE_HAKIRA = Data.groupby('Rank')['HAVE_HAKIRA'].apply(aggregations).reset_index()
# AggregationTable_HAVE_HAKIRA = AggregationTable_HAVE_HAKIRA.drop(columns=['level_1'])

# # Aggregation Suspicious_Money
# AggregationTable_Suspicious_HAVE_TVIA = Data.groupby('Rank')['HAVE_TVIA'].apply(aggregations).reset_index()
# AggregationTable_Suspicious_HAVE_TVIA = AggregationTable_Suspicious_HAVE_TVIA.drop(columns=['level_1'])


#### -------------------------------- Example 2 -------------------------------
#A = ["Si","Si","No","No","No"]
#B = ["Si","No","Si","Si","Si"]
#C = ["S?","No","S?","S?","No"]
#D = ["S?","S?","No","S?","S?"]

#Data = pd.DataFrame([A,B,C,D]).T
#Data.columns = ["A","B","C","D"]
#F = riditCalc(Data)
#Winf = iteractions(F)
#Compr_PCA = compPrinc(F)
#vectC = matBVectC(Data,Winf)
#print(vectC)

#conf = {
    #'UsingFactor': 'OnlyVariables',  ##Both, OnlyVariables, None
    #'FactorVariables': FactorVariables,  ##List, None
    #'NumericVariables': NumericVariables,  ##list, None
    #'FactorVariables': None,  ##List, None
    #'NumericVariables': None,  ##list, None
    #'FactorsVariablesOrder': None,  ##List, None
    #'NumericVariablesOrder': None  ##List, None
#}
#PriditClassifier = PriditClassifier(Data, conf={})
#priditScore,F,firstEigenVector = PriditClassifier.Pridit()
