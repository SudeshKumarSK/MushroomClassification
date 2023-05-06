##############################################################################
## EE559 Final Project ===> Mushroom Classification.
## Created by Sudesh Kumar Santhosh Kumar and Thejesh Chandar Rao.
## Date: 6th May, 2023
## Tested in Python 3.10.9 using conda environment version 22.9.0.
##############################################################################

import pandas as pd
import copy


def StatisticalEncoding(dataFrame, categoricalFeatures, numericalFeatures): 

    temp_df = copy.deepcopy(dataFrame)
    stats = ['mean', 'min', 'max', 'median']

    # Loop over all combinations of categorical and numerical features and calculate statistics
    for cat_feature in categoricalFeatures:
        for num_feature in numericalFeatures:
            for stat in stats:
                # Calculate the statistic for each group defined by the categorical feature
                group_stat = temp_df.groupby(cat_feature)[num_feature].agg(stat)
                # Create a new feature name based on the categorical feature and the statistic
                new_feature_name = cat_feature + '_' + num_feature + '_' + stat
                # Map the new feature to the data frame
                dataFrame[new_feature_name] = dataFrame[cat_feature].map(group_stat)

    # Drop the original categorical and numerical features
    temp_df.drop(categoricalFeatures, axis=1, inplace=True)

    return temp_df


def FrequencyEncoding():
    pass
