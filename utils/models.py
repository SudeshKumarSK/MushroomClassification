##############################################################################
## EE559 Final Project ===> Mushroom Classification.
## Created by Sudesh Kumar Santhosh Kumar and Thejesh Chandar Rao.
## Date: 7th May, 2023
## Tested in Python 3.10.9 using conda environment version 22.9.0.
##############################################################################


# Importing all necessary libraries.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import shutil


from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.kernel_approximation import RBFSampler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


class Model():
    def __init__(self):
        # Initializing the KFold as 4 and creating an instance for sklearn's KFold  
        self.kf = KFold(n_splits=5)

    
    def shuffleData(self, X_train, y_train):
        X_train_shuff, y_train_shuff = shuffle(X_train, y_train)

        return X_train_shuff, y_train_shuff
    

    def trivial_system(self, Y_true):

        total_num_points = Y_true.shape[0]
        counts = np.unique(Y_true, return_counts=True)
        num_points_p = int(counts[1][np.where(counts[0] == 1.0)])
        num_points_e = int(counts[1][np.where(counts[0] == 0.0)])
        prob_p = num_points_p/total_num_points
        prob_e = num_points_e/total_num_points
        predictions = np.random.choice([1.0, 0.0], size=total_num_points, p=[prob_p, prob_e])

        num_correct = np.sum(Y_true == predictions)
        acc = (num_correct/Y_true.shape[0])*100

        return acc
    

    def Baseline_System(self, X, y):
        
        # Split the data into 80% training and 20% validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = NearestCentroid()

        model = clf.fit(X_train, y_train)

        # Performing the same steps for the validation fold.
        Y_pred_train = clf.predict(X_train)

        # Performing the same steps for the validation fold.
        Y_pred_val = clf.predict(X_val)
        
        train_acc = accuracy_score(Y_pred_train, y_train)
        val_acc = accuracy_score(Y_pred_val, y_val)


        cer_val = 1 - val_acc
        cer_train = 1 - train_acc

        return (train_acc, val_acc)

    
    def testRun(self, X, y):

        # Split the data into 80% training and 20% validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = SVC(kernel="rbf")

        model = clf.fit(X_train, y_train)

        # Performing the same steps for the validation fold.
        Y_pred_train = clf.predict(X_train)

        # Performing the same steps for the validation fold.
        Y_pred_val = clf.predict(X_val)
        
        train_acc = accuracy_score(Y_pred_train, y_train)
        val_acc = accuracy_score(Y_pred_val, y_val)


        cer_val = 1 - val_acc
        cer_train = 1 - train_acc

        return (train_acc, val_acc)
    


    def train_Perceptron(self, X, y):

        # Split the data into 80% training and 20% validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = Perceptron(tol=1e-3, random_state=0)

        model = clf.fit(X_train, y_train)

        # Performing the same steps for the validation fold.
        Y_pred_train = clf.predict(X_train)

        # Performing the same steps for the validation fold.
        Y_pred_val = clf.predict(X_val)
        
        train_acc = accuracy_score(Y_pred_train, y_train)
        val_acc = accuracy_score(Y_pred_val, y_val)


        cer_val = 1 - val_acc
        cer_train = 1 - train_acc

        return (train_acc, val_acc)
    

    def train_SVM(self, X, y):

        # Split the data into 80% training and 20% validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = SVC(kernel="rbf")

        model = clf.fit(X_train, y_train)

        # Performing the same steps for the validation fold.
        Y_pred_train = clf.predict(X_train)

        # Performing the same steps for the validation fold.
        Y_pred_val = clf.predict(X_val)
        
        train_acc = accuracy_score(Y_pred_train, y_train)
        val_acc = accuracy_score(Y_pred_val, y_val)


        cer_val = 1 - val_acc
        cer_train = 1 - train_acc

        return (train_acc, val_acc)
    

    def train_KNN(self, X, y):

        # Split the data into 80% training and 20% validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = KNeighborsClassifier()

        model = clf.fit(X_train, y_train)

        # Performing the same steps for the validation fold.
        Y_pred_train = clf.predict(X_train)

        # Performing the same steps for the validation fold.
        Y_pred_val = clf.predict(X_val)
        
        train_acc = accuracy_score(Y_pred_train, y_train)
        val_acc = accuracy_score(Y_pred_val, y_val)


        cer_val = 1 - val_acc
        cer_train = 1 - train_acc

        return (train_acc, val_acc)
    

    def train_BayesClf(self, X, y):

        # Split the data into 80% training and 20% validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = GaussianNB()

        model = clf.fit(X_train, y_train)

        # Performing the same steps for the validation fold.
        Y_pred_train = clf.predict(X_train)

        # Performing the same steps for the validation fold.
        Y_pred_val = clf.predict(X_val)
        
        train_acc = accuracy_score(Y_pred_train, y_train)
        val_acc = accuracy_score(Y_pred_val, y_val)


        cer_val = 1 - val_acc
        cer_train = 1 - train_acc

        return (train_acc, val_acc)
    

    def train_MLP(self, X, y):

        # Split the data into 80% training and 20% validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


        mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='sgd', random_state=42)
        model = mlp.fit(X_train, y_train)

        # Performing the same steps for the validation fold.
        Y_pred_train = model.predict(X_train)

        # Performing the same steps for the validation fold.
        Y_pred_val = model.predict(X_val)
        
        train_acc = accuracy_score(Y_pred_train, y_train)
        val_acc = accuracy_score(Y_pred_val, y_val)


        cer_val = 1 - val_acc
        cer_train = 1 - train_acc

        return (train_acc, val_acc)



    def train_KNN_gridsearch(self, X_train, Y_train, runs = 1):
        model_Dict = {}
        run_val_CER = []

        run_mean_val_CER = 0
        run_std_val_CER = 0

        run_minimum_val_CER = 0
        run_maximum_val_CER = 0

        run_max = 0
        run_min = 0

        for run in range(runs):

            val_CER = []
            mean_val_CER = 0

            X_train_shuffled, y_train_shuffled = self.shuffleData(X_train, Y_train)
            for i, (train_index, val_index) in enumerate(self.kf.split(X_train)):

                # Getting the folds of train and validation data one by one 20 times in this loop.
                # Basically X_train_fold will contain n_train/20 data-points in one iteration and X_val_fold will had the rest 1000 data-points.
                X_train_fold, X_val_fold = X_train_shuffled[train_index], X_train_shuffled[val_index]
                Y_train_fold, Y_val_fold = y_train_shuffled[train_index], y_train_shuffled[val_index]

                clf = SVC(kernel="rbf", C=0.01, gamma=10)

                model = clf.fit(X_train_fold, Y_train_fold)

                # Performing the same steps for the validation fold.
                Y_pred_val_fold = clf.predict(X_val_fold)
                
                val_acc = accuracy_score(Y_pred_val_fold, Y_val_fold)
                cer = 1 - val_acc

                val_CER.append(cer)

                if i == 0:
                    model_Dict[run] = model

                elif cer < val_CER[-1]:
                    model_Dict[run] = model


            mean_val_CER = np.mean(val_CER)
            run_val_CER.append(mean_val_CER)

        run_mean_val_CER = np.mean(run_val_CER)
        run_std_val_CER = np.std(run_val_CER)

        run_maximum_val_CER = np.max(run_val_CER)
        run_minimum_val_CER = np.min(run_val_CER)

        run_max = np.argmax(run_val_CER)
        run_min = np.argmin(run_val_CER)

        print("-----------------------------------------------------------------------------------------------------------------------------------")
        # print(f"The Mean Classification Error Rate from each cross-val run are: {dict(zip([1, 2, 3, 4, 5], run_val_CER))}")
        print(f"The Average of the Mean Classification Error Rate over the 5 runs is: {run_mean_val_CER}")
        print(f"The Standard Deviation of the Mean Classification Error Rate over the 5 runs is: {run_std_val_CER}")
        print(f"The Lowest Mean Classification Error Rate -> {run_minimum_val_CER} was achieved at run: {run_min + 1}")
        print(f"The Higesh Mean Classification Error Rate -> {run_maximum_val_CER} was achieved at run: {run_max + 1}")
        print("-----------------------------------------------------------------------------------------------------------------------------------")

        model_min_CER = model_Dict[run_min]
        model_max_CER = model_Dict[run_max]

        return (run_max, run_min, model_min_CER, model_max_CER)
    

    def train_Different_Data(self, X_train_raw, X_train_LDA, X_train_PCA_15, X_train_PCA_30, X_train_PCA_45, X_train_PCA_60, X_train_PCA_90, y_train, model_name):
 
        result = {}

        if model_name == "baseline":

            result["raw"] = self.Baseline_System(X = X_train_raw, y = y_train)
            print(f"Trained the {model_name} model on raw data successfully!")

            result["LDA"] = self.Baseline_System(X = X_train_LDA, y = y_train)
            print(f"Trained the {model_name} model on LDA transformed data successfully!")

            result["PCA_15"] = self.Baseline_System(X = X_train_PCA_15, y = y_train)
            print(f"Trained the {model_name} model on PCA_15 transformed data successfully!")

            result["PCA_30"] = self.Baseline_System(X = X_train_PCA_30, y = y_train)
            print(f"Trained the {model_name} model on PCA_30 transformed data successfully!")

            result["PCA_45"] = self.Baseline_System(X = X_train_PCA_45, y = y_train)
            print(f"Trained the {model_name} model on PCA_45 transformed data successfully!")

            result["PCA_60"] = self.Baseline_System(X = X_train_PCA_60, y = y_train)
            print(f"Trained the {model_name} model on PCA_60 transformed data successfully!")

            result["PCA_90"] = self.Baseline_System(X = X_train_PCA_90, y = y_train)
            print(f"Trained the {model_name} model on PCA_90 transformed data successfully!")


        elif model_name == "perceptron":
            result["raw"] = self.train_Perceptron(X = X_train_raw, y = y_train)
            print(f"Trained the {model_name} model on raw data successfully!")

            result["LDA"] = self.train_Perceptron(X = X_train_LDA, y = y_train)
            print(f"Trained the {model_name} model on LDA transformed data successfully!")

            result["PCA_15"] = self.train_Perceptron(X = X_train_PCA_15, y = y_train)
            print(f"Trained the {model_name} model on PCA_15 transformed data successfully!")

            result["PCA_30"] = self.train_Perceptron(X = X_train_PCA_30, y = y_train)
            print(f"Trained the {model_name} model on PCA_30 transformed data successfully!")

            result["PCA_45"] = self.train_Perceptron(X = X_train_PCA_45, y = y_train)
            print(f"Trained the {model_name} model on PCA_45 transformed data successfully!")

            result["PCA_60"] = self.train_Perceptron(X = X_train_PCA_90, y = y_train)
            print(f"Trained the {model_name} model on PCA_60 transformed data successfully!")

            result["PCA_90"] = self.train_Perceptron(X = X_train_PCA_90, y = y_train)
            print(f"Trained the {model_name} model on PCA_90 transformed data successfully!")


        elif model_name == "SVM":
            result["raw"] = self.train_SVM(X = X_train_raw, y = y_train)
            print(f"Trained the {model_name} model on raw data successfully!")

            result["LDA"] = self.train_SVM(X = X_train_LDA, y = y_train)
            print(f"Trained the {model_name} model on LDA transformed data successfully!")

            result["PCA_15"] = self.train_SVM(X = X_train_PCA_15, y = y_train)
            print(f"Trained the {model_name} model on PCA_15 transformed data successfully!")

            result["PCA_30"] = self.train_SVM(X = X_train_PCA_30, y = y_train)
            print(f"Trained the {model_name} model on PCA_30 transformed data successfully!")

            result["PCA_45"] = self.train_SVM(X = X_train_PCA_45, y = y_train)
            print(f"Trained the {model_name} model on PCA_45 transformed data successfully!")

            result["PCA_60"] = self.train_SVM(X = X_train_PCA_60, y = y_train)
            print(f"Trained the {model_name} model on PCA_60 transformed data successfully!")

            result["PCA_90"] = self.train_SVM(X = X_train_PCA_90, y = y_train)
            print(f"Trained the {model_name} model on PCA_90 transformed data successfully!")


        elif model_name == "KNN":
            result["raw"] = self.train_KNN(X = X_train_raw, y = y_train)
            print(f"Trained the {model_name} model on raw data successfully!")

            result["LDA"] = self.train_KNN(X = X_train_LDA, y = y_train)
            print(f"Trained the {model_name} model on LDA transformed data successfully!")

            result["PCA_15"] = self.train_KNN(X = X_train_PCA_15, y = y_train)
            print(f"Trained the {model_name} model on PCA_15 transformed data successfully!")

            result["PCA_30"] = self.train_KNN(X = X_train_PCA_30, y = y_train)
            print(f"Trained the {model_name} model on PCA_30 transformed data successfully!")

            result["PCA_45"] = self.train_KNN(X = X_train_PCA_45, y = y_train)
            print(f"Trained the {model_name} model on PCA_45 transformed data successfully!")

            result["PCA_60"] = self.train_KNN(X = X_train_PCA_60, y = y_train)
            print(f"Trained the {model_name} model on PCA_60 transformed data successfully!")

            result["PCA_90"] = self.train_KNN(X = X_train_PCA_90, y = y_train)
            print(f"Trained the {model_name} model on PCA_90 transformed data successfully!")


        elif model_name == "BayesClf":
            result["raw"] = self.train_BayesClf(X = X_train_raw, y = y_train)
            print(f"Trained the {model_name} model on raw data successfully!")

            result["LDA"] = self.train_BayesClf(X = X_train_LDA, y = y_train)
            print(f"Trained the {model_name} model on LDA transformed data successfully!")

            result["PCA_15"] = self.train_BayesClf(X = X_train_PCA_15, y = y_train)
            print(f"Trained the {model_name} model on PCA_15 transformed data successfully!")

            result["PCA_30"] = self.train_BayesClf(X = X_train_PCA_30, y = y_train)
            print(f"Trained the {model_name} model on PCA_30 transformed data successfully!")

            result["PCA_45"] = self.train_BayesClf(X = X_train_PCA_45, y = y_train)
            print(f"Trained the {model_name} model on PCA_45 transformed data successfully!")

            result["PCA_60"] = self.train_BayesClf(X = X_train_PCA_60, y = y_train)
            print(f"Trained the {model_name} model on PCA_60 transformed data successfully!")

            result["PCA_90"] = self.train_BayesClf(X = X_train_PCA_90, y = y_train)
            print(f"Trained the {model_name} model on PCA_90 transformed data successfully!")
            
        else:
            result["raw"] = self.train_MLP(X = X_train_raw, y = y_train)
            print(f"Trained the {model_name} model on raw data successfully!")

            result["LDA"] = self.train_MLP(X = X_train_LDA, y = y_train)
            print(f"Trained the {model_name} model on LDA transformed data successfully!")

            result["PCA_15"] = self.train_MLP(X = X_train_PCA_15, y = y_train)
            print(f"Trained the {model_name} model on PCA_15 transformed data successfully!")

            result["PCA_30"] = self.train_MLP(X = X_train_PCA_30, y = y_train)
            print(f"Trained the {model_name} model on PCA_30 transformed data successfully!")

            result["PCA_45"] = self.train_MLP(X = X_train_PCA_45, y = y_train)
            print(f"Trained the {model_name} model on PCA_45 transformed data successfully!")

            result["PCA_60"] = self.train_MLP(X = X_train_PCA_60, y = y_train)
            print(f"Trained the {model_name} model on PCA_60 transformed data successfully!")

            result["PCA_90"] = self.train_MLP(X = X_train_PCA_90, y = y_train)
            print(f"Trained the {model_name} model on PCA_90 transformed data successfully!")


        return result


    def final_train(self, X_train, y_train, model_name):

        curr_model = None
        train_acc = 0.0
        train_cer = 0.0

        ## Baseline Model
        if model_name == "baseline":
            clf = NearestCentroid()

            curr_model = clf.fit(X_train, y_train)

            # Performing the same steps for the validation fold.
            Y_pred = clf.predict(X_train)
            
            train_acc = accuracy_score(Y_pred, y_train)

            train_cer = 1 - train_acc
        
        ## Non-Probablistic Models {Perceptron and KNN}

        elif model_name == "perceptron":

            clf = Perceptron(tol=1e-3, random_state=0)

            curr_model = clf.fit(X_train, y_train)

            # Performing the same steps for the validation fold.
            Y_pred = clf.predict(X_train)
            
            train_acc = accuracy_score(Y_pred, y_train)

            train_cer = 1 - train_acc


        elif model_name == "KNN":

            clf = KNeighborsClassifier()

            curr_model = clf.fit(X_train, y_train)

            # Performing the same steps for the validation fold.
            Y_pred = clf.predict(X_train)
            
            train_acc = accuracy_score(Y_pred, y_train)

            train_cer = 1 - train_acc

        # Probabilitic Approach Bayes Minimum Error Classifier.
        elif model_name == "BayesClf":
            clf = GaussianNB()

            curr_model = clf.fit(X_train, y_train)

            # Performing the same steps for the validation fold.
            Y_pred = clf.predict(X_train)
            
            train_acc = accuracy_score(Y_pred, y_train)

            train_cer = 1 - train_acc

        
        # Support Vector Machine.
        elif model_name == "SVM":
            clf = SVC(kernel = "rbf")

            curr_model = clf.fit(X_train, y_train)

            # Performing the same steps for the validation fold.
            Y_pred = clf.predict(X_train)
            
            train_acc = accuracy_score(Y_pred, y_train)

            train_cer = 1 - train_acc


        # A  two later MLP using relu actiavtion function and .

        else:
            mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='sgd', random_state=42)
            curr_model = mlp.fit(X_train, y_train)

            # Performing the same steps for the validation fold.
            Y_pred = curr_model.predict(X_train)
        
            train_acc = accuracy_score(Y_pred, y_train)

            train_cer = 1 - train_acc


        # Save the model using pickle
        with open(f"{model_name}.pkl", "wb") as model_file:
            pickle.dump(curr_model, model_file)

        # Save the model using pickle
        with open(f"{model_name}_backup.pkl", "wb") as model_file:
            pickle.dump(curr_model, model_file)

        # Specify the current file path
        current_file_path = f"./{model_name}.pkl"
        # Specify the destination folder path
        destination_folder_path = f"/Users/sk/ee559-mlOne/MushroomClassification/saved_models/{model_name}.pkl"
        # Move the file to the destination folder
        shutil.move(current_file_path, destination_folder_path)

        # Specify the current file path
        current_file_path = f"./{model_name}_backup.pkl"
        # Specify the destination folder path
        destination_folder_path = f"./models_backup/{model_name}.pkl"
        # Move the file to the destination folder
        shutil.move(current_file_path, destination_folder_path)

        return (train_acc * 100, train_cer * 100)
    
            