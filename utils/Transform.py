##############################################################################
## EE559 Final Project ===> Mushroom Classification.
## Created by Sudesh Kumar Santhosh Kumar and Thejesh Chandar Rao.
## Date: 6th May, 2023
## Tested in Python 3.10.9 using conda environment version 22.9.0.
##############################################################################


#Importing all necessary libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SequentialFeatureSelector as SFS
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

    X_std_df = pd.DataFrame(X_std, columns=X.columns) 

    return X_std_df


# User-defined Function to perform PCA on the input Features.
def transformTrainData_PCA(X, components):
    pca = sklearnPCA(n_components = components)
    X_PCA = pca.fit_transform(X) 

    return X_PCA    


# User-defined Function to perform LDA on the input Features.
def transformTrainData_LDA(X, Y):
    lda = LDA(n_components = 1)
    X_LDA = lda.fit_transform(X, Y)

    return X_LDA

# User-defined Function to perform LDA on the input Features.
def transformTrainData_IterLDA(X, Y, components):
    lda = LDA()
    # Define SFS algorithm
    sfs = SFS(lda, n_features_to_select=components, direction='forward', cv=10)

    # Fit SFS algorithm to data
    sfs.fit(X, Y)

    # Transform data to selected features
    X_sfs = sfs.transform(X)

    return X_sfs

