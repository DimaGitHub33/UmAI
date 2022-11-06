## F calculation for Factor variables  ------------------------------------
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pd.options.display.float_format = '{:.2f}'.format

def riditCalc(Data,factorsVariablesOrder=None):
    F = pd.DataFrame()
    for variableToConvert in Data.columns:
        #print(variableToConvert)
        variable = Data[[variableToConvert]].copy()
        variable.columns = ["variableToConvert"]
        variable.loc[:, 'variableToConvert'] = variable['variableToConvert'].astype(str).fillna('NULL')

        # Frequency table
        if (len(variable['variableToConvert'].unique()) < 2):
            continue

        frequencyTable = pd.DataFrame(variable['variableToConvert'].value_counts(normalize=True)).reset_index()
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

    return F

def iteractions(F):

    Ft = F.T
    W0 = np.ones((1,len(Ft)), dtype=np.float32)
    S0 = np.dot(F,W0.T)
    W1a = np.dot(Ft,S0)
    W1b = np.linalg.norm(W1a)
    W1 = W1a/W1b
    S1 = np.dot(F,W1) 

    iter = 1
    while iter<=1000:
        iter = iter+1
        Wa = np.dot(Ft,S1)
        Wb = np.linalg.norm(Wa)
        W2 = Wa/Wb
        S2 = np.dot(F,W2) 
        W0 = W1
        W1 = W2
        S1 = S2

    return W2


def compPrinc(F):
    # Apply PCA
    pca = PCA(n_components=1)
    X = pca.fit_transform(-F,)

    loadings = pd.DataFrame(pca.components_.T, columns=['PC1'], index=F.columns)
    return loadings.PC1


def matBVectC(Data,Winf):
    Weights = {}
    factorsVariablesOrder = None
    for variableToConvert in Data.columns:
        #print(variableToConvert)
        variable = Data[[variableToConvert]].copy()
        variable.columns = ["variableToConvert"]
        variable.loc[:, 'variableToConvert'] = variable['variableToConvert'].astype(str).fillna('NULL')

        # Frequency table
        if (len(variable['variableToConvert'].unique()) < 2):
            continue

        frequencyTable = pd.DataFrame(variable['variableToConvert'].value_counts(normalize=True)).reset_index()
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

        Weights[variableToConvert] = [-frequencyTable.iloc[1,1],frequencyTable.iloc[0,1]]


    matB = np.diag(np.full(Data.shape[1]*2,1.0))  
    auxcol = list(Weights.values())[0]
    cont = 0
    for j in range(0,len(auxcol)):
        for i in range(0,len(Weights)):
            matB[cont,cont] = list(Weights.values())[i][j] 
            cont = cont + 1	

    print("Matrix B")
    print(matB)
    
    print("Vector Winf")
    print(Winf)

    Winf2 = np.zeros(len(Winf)*2).T
    for i in range(0,len(Winf)):
        Winf2[i] = Winf[i][0]
        Winf2[i+len(Winf)] = Winf[i]

    vectorC = np.dot(matB,Winf2)

    return vectorC





A = ["Si","Si","No","No","No"]
B = ["Si","No","Si","Si","Si"]
C = ["S?","No","S?","S?","No"]
D = ["S?","S?","No","S?","S?"]

x1 = pd.DataFrame([A,B,C,D]).T
x1.columns = ["A","B","C","D"]
Data = x1
F = riditCalc(Data)
Winf = iteractions(F)
Compr_PCA = compPrinc(F)
vectC = matBVectC(Data,Winf)
print(vectC)
