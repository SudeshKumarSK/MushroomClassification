##############################################################################
## EE559 Final Project ===> Mushroom Classification.
## Created by Sudesh Kumar Santhosh Kumar and Thejesh Chandar Rao.
## Date: 6th May, 2023
## Tested in Python 3.10.9 using conda environment version 22.9.0.
##############################################################################

import pandas as pd
import copy


def StatisticalEncoding(dataFrame, categoricalFeatures, numericalFeatures): 

    temp_df = dataFrame[::]

    X_train_temp = temp_df.drop('class', axis=1)  # Select all the features except labels,
    y_train_temp = temp_df['class']  # Select only the 'class' column.
    stats = ['mean', 'min', 'max', 'median']

    # Loop over all combinations of categorical and numerical features and calculate statistics
    for cat_feature in categoricalFeatures:
        for num_feature in numericalFeatures:
            for stat in stats:
                # Calculate the statistic for each group defined by the categorical feature
                group_stat = X_train_temp.groupby(cat_feature)[num_feature].agg(stat)
                # Create a new feature name based on the categorical feature and the statistic
                new_feature_name = cat_feature + '_' + num_feature + '_' + stat
                # Map the new feature to the data frame
                # X_train_temp[new_feature_name] = X_train_temp[cat_feature].map(group_stat)

                new_column = pd.DataFrame({new_feature_name: X_train_temp[cat_feature].map(group_stat)})

                X_train_temp = pd.concat([X_train_temp, new_column], axis=1)

    # Drop the original categorical and numerical features
    X_train_temp.drop(categoricalFeatures, axis=1, inplace=True)

    temp_df_final = pd.concat([X_train_temp, y_train_temp], axis=1)

    return temp_df_final


def OneHotEncoding(dataFrame, categoricalFeatures, numericalFeatures):
    temp_df = dataFrame[::]

    X_train_temp = temp_df.drop('class', axis=1)  # Select all the features except labels,
    y_train_temp = temp_df['class']  # Select only the 'class' column.

    for cat_feature in categoricalFeatures:
        X_train_temp_enc = pd.get_dummies(X_train_temp[cat_feature], prefix=cat_feature)
        X_train_temp = pd.concat([X_train_temp, X_train_temp_enc], axis=1)
        X_train_temp.drop(cat_feature, axis=1, inplace=True)

    temp_df_final = pd.concat([X_train_temp, y_train_temp], axis=1)

    return temp_df_final
        
