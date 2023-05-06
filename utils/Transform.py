##############################################################################
## EE559 Final Project ===> Mushroom Classification.
## Created by Sudesh Kumar Santhosh Kumar and Thejesh Chandar Rao.
## Date: 6th May, 2023
## Tested in Python 3.10.9 using conda environment version 22.9.0.
##############################################################################


#Importing all necessary libraries
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler



# User-defined Function to Standardize the numpy Data passed.
def standardizeData(X):

    '''
        Preprocessing the train data using Scikit-learn.
    '''

    scaler = StandardScaler()
    scaler.fit(X)
    X_std = scaler.transform(X)
    print("Standardized the Train Data!")

    return X_std


# User-defined Function to perform PCA on the input Features.
def transformTrainData_PCA(X, components):
    pca = sklearnPCA(n_components = components)
    X_PCA = pca.fit_transform(X) 

    return X_PCA    


# User-defined Function to perform LDA on the input Features.
def transformTrainData_LDA(X, Y, components):
    lda = LDA(n_components = 2)
    X_LDA = lda.fit_transform(X, Y)

    return X_LDA

