Contributors:
Sudesh Kumar Santhosh Kumar, santhosh@usc.edu
Thejesh Chandar Rao Jadav Chandrasekar Rao, jadavcha@usc.edu

Project Structure:

All the necessary helper functions to perform Encoding like Statistical Encoding, One-Hot Encoding are present in the Encoding.py file.
All the Dimensionality Reduction Techniques like PCA and LDA are included in the Transform.py file. In addition to these files, all the models used in this project are coded and kept inside the models.py file.

/utils:
Encoding.py
Transform.py
models.py

These three helper functions are present inside the /utils directory.

There are in total 6 python notebook files.

1. FeatureEngineering.ipynb -> Contains all the feature engineering steps
2. FeatureSelection_OneHot.ipynb -> Feature Selection using training simple linear models and retaining highlt correlated features are present in this file.
3. FeatureSelection_Stat.ipynb -> Feature Selection using training simple linear models and retaining highlt correlated features are present in this file.
4. FeatureTransformation -> Reduction of Dimensions in the Selected FEatures to improve performance is present in this file.
5. Model Selection -> Performing cross-validation, tuning of hyperparameters, selecting the feature engineering technique which produces better performances are all performed in this file and the best technique is used in the FinalTrainTest.ipynb
6. FinalTrainTest.ipynb -> Final complete training of data and Testing the models is performed here.

/dataset
All the datasets used are present here.

/plots
All the necessary plots are present here in the file.

/saved_models
All the trained models are pickled and saved in this directory.

/models_backup
All the trained models are pickled and saved in this directory and used as a backup for safety.

Additional Libraries Used:

1. Seaborn
2. pickle
3. imshutil
