import pandas as pd
import numpy as np
import os, random, joblib, statistics
import sklearn as sk
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GroupKFold
import functions
random.seed(1213870)

path1 = os.path.join('data', 'speech_data')

# Load training + validation sets (male, female mixed) >> load datasets with cross-validation folds!!
data_q_male_train = pd.read_csv(os.path.join(path1,'data_data_q_male.csv'))
data_q_female_train = pd.read_csv(os.path.join(path1,'data_data_q_female.csv'))
data_q_mixed_train = pd.read_csv(os.path.join(path1,'data_data_q_mixed.csv'))

# Load test sets (male, female, mixed) 
data_q_male_test = pd.read_csv(os.path.join(path1,'data_q_male_test.csv'))
data_q_female_test = pd.read_csv(os.path.join(path1,'data_q_female_test.csv'))
data_q_mixed_test = pd.read_csv(os.path.join(path1,'data_q_mixed_test.csv'))


#Path for getting feature lists (male, female, mixed) 
path2 = os.path.join('data', 'feature_lists')

# get stable features (present in 2+ feature selection folds)
n_folds = 3
stable_df_male = pd.read_csv("data/feature_lists/features_data_q_male_STABLE.csv")
stable_features_male = stable_df_male["features"].str.strip("`").tolist()
features_male = [stable_features_male for _ in range(n_folds)]

stable_df_female = pd.read_csv("data/feature_lists/features_data_q_female_STABLE.csv")
stable_features_female = stable_df_female["features"].str.strip("`").tolist()
features_female = [stable_features_female for _ in range(n_folds)]

stable_df_mixed = pd.read_csv("data/feature_lists/features_data_q_mixed_STABLE.csv")
stable_features_mixed = stable_df_mixed["features"].str.strip("`").tolist() 
features_mixed = [stable_features_mixed for _ in range(n_folds)]

if __name__=='__main__':
    # Train and tweak hyperparameters
    save = True
    output = functions.train_tweak_hyper(train=data_q_male_train, train_name="model_male", feature_lists=features_male, kernel = "rbf", save=save) # male model
    output = functions.train_tweak_hyper(train=data_q_female_train, train_name="model_female", feature_lists=features_female, kernel = "rbf", save=save) # female model
    output = functions.train_tweak_hyper(train=data_q_mixed_train, train_name="model_mixed", feature_lists=features_mixed, kernel = "rbf", save=save) # mixed model

    # Testing using male model
    output = functions.test_model(train_name="model_male", test_name="male", kernel="rbf", save=save, test=data_q_male_test, feature_lists=features_male)
    output = functions.test_model(train_name="model_male", test_name="female", kernel="rbf", save=save, test=data_q_female_test, feature_lists=features_male)
    output = functions.test_model(train_name="model_male", test_name="mixed", kernel="rbf", save=save, test=data_q_mixed_test, feature_lists=features_male)
    
    # Testing using female model
    output = functions.test_model(train_name="model_female", test_name="male", kernel="rbf", save=save, test=data_q_male_test, feature_lists=features_female)
    output = functions.test_model(train_name="model_female", test_name="female", kernel="rbf", save=save, test=data_q_female_test, feature_lists=features_female)
    output = functions.test_model(train_name="model_female", test_name="mixed", kernel="rbf", save=save, test=data_q_mixed_test, feature_lists=features_female)

    # Testing using mixed model
    output = functions.test_model(train_name="model_mixed", test_name="male", kernel="rbf", save=save, test=data_q_male_test, feature_lists=features_mixed)
    output = functions.test_model(train_name="model_mixed", test_name="female", kernel="rbf", save=save, test=data_q_female_test, feature_lists=features_mixed)
    output = functions.test_model(train_name="model_mixed", test_name="mixed", kernel="rbf", save=save, test=data_q_mixed_test, feature_lists=features_mixed)


functions.write_selected_features(features_male, "male")
functions.write_selected_features(features_female, "female")
functions.write_selected_features(features_mixed, "mixed")
