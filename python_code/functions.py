from typing import List, Optional, Tuple
import os, joblib
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# ------------------------------------------------------------------
# 1)  FEATURE-LIST HELPER
# ------------------------------------------------------------------
def get_feature_list(path: str,
                     file_name: str,
                     n_folds: Optional[int] = 3) -> List[List[str]]:
    """Takes a file containing cross-validated feature selection.
    Extracts selected features and returns a list of lists with the features

    Args:
        path (str): path to folder with file
        file_name (str): file name (including .csv)
        n_folds (int, optional): number of cross-validation folds. Defaults to 3.

    Returns:
        List[List[str]]: List of n_folds feature lists
    """
    feature_df = pd.read_csv(os.path.join(path, file_name))   # keep both columns
    feature_list = []
    for i in range(n_folds):
        single_set = feature_df.loc[feature_df["fold"] == i + 1, "features"].tolist()
        feature_list.append(single_set)                       # can be empty
    return feature_list

# ------------------------------------------------------------------
# 2)  TRAIN + INNER VALIDATION
# ------------------------------------------------------------------
def train_tweak_hyper(train: pd.DataFrame,
                      train_name: str,
                      feature_lists: List[List[str]],
                      kernel: str,
                      save: bool,
                      gamma: str = "scale"):
    """This function is for training SVM with specified kernel and tweaking
    hyperparameters on the validation set.

    Args:
        train (pd.DataFrame): Full train data set with all features and a column
                              indicating cross-validation folds
        train_name (str): String to use for folder name and file pre-name
        feature_lists (List[List[str]]): List of feature lists with strings
                                         indicating selected features
        kernel (str): Takes values 'linear', 'poly', 'rbf', 'sigmoid' or 'precomputed'
        save (bool): True or False – whether or not to save classification reports,
                     model predictions and confusion matrices

    Returns:
        tuple: containing validation classification reports,
               validation confusion matrices, validation model predictions
    """
    # Empty lists for appending
    validation_classification_reports = []
    validation_confusion_matrices = []
    validation_model_predictions = []

    # Define parameter grid for GridSearchCV
    # Exponentially scaled parameter ranges (as per the paper)
    C_values = [0.01, 0.1, 0.5, 1, 5, 10, 50, 100]
    gamma_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

    param_grid = {
    'C': C_values,
    'gamma': gamma_values,
    'kernel': [kernel],
    'class_weight': ["balanced"],
    }   

    index_list = list(range(1, len(feature_lists) + 1))

    for n in index_list:
        if len(feature_lists[n - 1]) == 0:
            print(f"Fold {n}: NO features selected – skipping this model.")
            validation_classification_reports.append(None)
            validation_confusion_matrices.append(None)
            validation_model_predictions.append(None)
            continue

        # For feature set n-1 model, subset training data to only include other folds
        train_subset = train.loc[train[".folds"] != n]
        validation = train.loc[train[".folds"] == n]

        # Dividing into predictor variables (x) and what should be predicted (y)
        trainX = train_subset.loc[:, feature_lists[n - 1]]
        trainY = train_subset["Diagnosis"]  # Changed to just the label
        validationX = validation.loc[:, feature_lists[n - 1]]
        validationY = validation["Diagnosis"]
        validationID = validation["ID"]

        # Set up and run GridSearchCV on the training data
        grid_search = GridSearchCV(
            SVC(),
            param_grid,
            scoring='f1_weighted',  
            cv=3, 
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(trainX, trainY)
        best_model = grid_search.best_estimator_
        print(f"Fold {n}: Best params: {grid_search.best_params_}")

        # Save the best model for this fold
        os.makedirs(f"../data/models/{train_name}", exist_ok=True)
        joblib.dump(best_model, f"../data/models/{train_name}/{kernel}_{n}.pkl")

        # Predict on validation set
        validation_predictions = best_model.predict(validationX)

        # Retrieve performance measures
        validation_classification_report = pd.DataFrame(
            classification_report(validationY, validation_predictions, output_dict=True))
        validation_confusion_matrix = pd.DataFrame(
            confusion_matrix(validationY, validation_predictions))

        validation_classification_reports.append(validation_classification_report)
        validation_confusion_matrices.append(validation_confusion_matrix)

        # Store predictions
        model_predictions = pd.DataFrame({
            f"fold_{n}_true_diagnosis": validationY,
            f"fold_{n}_predicted_diagnosis": validation_predictions,
            f"ID_{n}": validationID
        })
        model_predictions["Correct"] = (
            model_predictions[f"fold_{n}_true_diagnosis"]
            == model_predictions[f"fold_{n}_predicted_diagnosis"]
        )
        validation_model_predictions.append(model_predictions)

    # Saving (same as before)
    if save:
        os.makedirs(f"results/validation/{train_name}", exist_ok=True)
        for n in index_list:
            if validation_classification_reports[n - 1] is None:
                continue
            validation_model_predictions[n - 1].to_csv(
                f"results/validation/{train_name}/{train_name}_{kernel}_model_predictions_{n}.csv")
            validation_classification_reports[n - 1].to_csv(
                f"results/validation/{train_name}/{train_name}_{kernel}_classification_report_{n}.csv")
            validation_confusion_matrices[n - 1].to_csv(
                f"results/validation/{train_name}/{train_name}_{kernel}_confusion_matrix_{n}.csv")

    return validation_classification_reports, validation_confusion_matrices, validation_model_predictions


# ------------------------------------------------------------------
# 3)  TEST ON HOLD-OUT
# ------------------------------------------------------------------
def test_model(train_name: str,
               test_name: str,
               kernel: str,
               save: bool,
               test: pd.DataFrame,
               feature_lists: List[List[str]]):
    """This function tests a trained model on a hold-out set to get out-of-sample
    performance

    Args:
        train_name (str): string to use for folder name and file pre-name – should match existing folder and file
        test_name  (str): string to use for test folder name and file pre-name when saving performance
        kernel     (str): string specifying kernel used for training model – used to extract correct model
        save       (bool): whether or not to save performance metrics
        test (pd.DataFrame): dataframe with the hold-out data
        feature_lists (List[List[str]]): list of lists of features selected

    Returns:
        tuple: classification reports, confusion matrices,
               model predictions, ensemble classification_report, ensemble confusion_matrix
    """
    # Empty lists for appending
    classification_reports = []
    confusion_matrices = []
    model_predictions = pd.DataFrame({"true_diagnosis": test["Diagnosis"]})

    # Creating a list of numbers from 1 to number of feature lists
    index_list = list(range(1, len(feature_lists) + 1))
    used_models = []   

    # Loop over models
    for n in index_list:

        if len(feature_lists[n - 1]) == 0:
            print(f"Fold {n}: no features – skipping.")
            continue
        model_path = f"../data/models/{train_name}/{kernel}_{n}.pkl"
        if not os.path.exists(model_path):
            print(f"Fold {n}: model file missing – skipping.")
            continue

        used_models.append(n)

        # Divide up the test set
        testX = test.loc[:, feature_lists[n - 1]]
        testY = test.loc[:, "Diagnosis"]

        # Predict test set with saved model
        predictions = joblib.load(model_path).predict(testX)

        # Retrieving performance measures
        classif_report = pd.DataFrame(
            classification_report(testY, predictions, output_dict=True))
        conf_matrix = pd.DataFrame(confusion_matrix(testY, predictions))

        # Loading the performance into the empty lists
        classification_reports.append(classif_report)
        confusion_matrices.append(conf_matrix)

        # Retrieving true diagnosis and model predictions and load it into dataframe
        model_predictions[f"model_{n}_predicted_diagnosis"] = predictions

    # Ensemble vote only if at least one model ran
    if used_models:
        prediction_cols = [f"model_{n}_predicted_diagnosis" for n in used_models]
        ensemble_predictions = model_predictions[prediction_cols].mode(axis=1)
        model_predictions["ensemble_predictions"] = ensemble_predictions.iloc[:, 0]
        model_predictions["ID"] = test["ID"]
        model_predictions["Correct"] = (
            model_predictions["true_diagnosis"] == model_predictions["ensemble_predictions"]
        )

        ensemble_classification_report = pd.DataFrame(
            classification_report(test["Diagnosis"],
                                  ensemble_predictions.iloc[:, 0],
                                  output_dict=True))
        ensemble_confusion_matrix = pd.DataFrame(
            confusion_matrix(test["Diagnosis"], ensemble_predictions.iloc[:, 0]))
        
        # Participant-level evaluation ---
        participant_preds = (
            model_predictions
            .groupby("ID")
            .agg({
                "true_diagnosis": "first",
                "ensemble_predictions": lambda x: x.mode()[0]
            })
            .reset_index()
        )
        participant_preds["Correct"] = (
            participant_preds["true_diagnosis"] == participant_preds["ensemble_predictions"]
        )

        participant_classification_report = pd.DataFrame(
            classification_report(participant_preds["true_diagnosis"],
                                  participant_preds["ensemble_predictions"],
                                  output_dict=True))
        participant_confusion_matrix = pd.DataFrame(
            confusion_matrix(participant_preds["true_diagnosis"],
                             participant_preds["ensemble_predictions"]))
    else:
        print("No usable models – ensemble not created.")
        ensemble_classification_report = None
        ensemble_confusion_matrix = None
        participant_classification_report = None
        participant_confusion_matrix = None
        participant_preds = None

    # Saving output 
    if save:
        os.makedirs(f"results/test/{train_name}", exist_ok=True)

        model_predictions.to_csv(
            f"results/test/{train_name}/{train_name}_tested_on_{test_name}_{kernel}_model_predictions.csv")

        if ensemble_classification_report is not None:
            ensemble_classification_report.to_csv(
                f"results/test/{train_name}/{train_name}_tested_on_{test_name}_{kernel}_classification_report_ensemble.csv")
            ensemble_confusion_matrix.to_csv(
                f"results/test/{train_name}/{train_name}_tested_on_{test_name}_{kernel}_confusion_matrix_ensemble.csv")

            # Save participant-level reports
            participant_preds.to_csv(
                f"results/test/{train_name}/{train_name}_tested_on_{test_name}_{kernel}_participant_predictions.csv")
            participant_classification_report.to_csv(
                f"results/test/{train_name}/{train_name}_tested_on_{test_name}_{kernel}_classification_report_participants.csv")
            participant_confusion_matrix.to_csv(
                f"results/test/{train_name}/{train_name}_tested_on_{test_name}_{kernel}_confusion_matrix_participants.csv")

        for idx, fold in enumerate(used_models, start=1):
            classification_reports[idx - 1].to_csv(
                f"results/test/{train_name}/{train_name}_tested_on_{test_name}_{kernel}_classification_report_{fold}.csv")
            confusion_matrices[idx - 1].to_csv(
                f"results/test/{train_name}/{train_name}_tested_on_{test_name}_{kernel}_confusion_matrix_{fold}.csv")
            
        print("\n=== ENSEMBLE CLASSIFICATION REPORT ===")
        print(f"Model: {train_name} | Tested on: {test_name}")
        print(participant_classification_report.transpose().round(2))

    return (
        classification_reports,
        confusion_matrices,
        model_predictions,
        ensemble_classification_report,
        ensemble_confusion_matrix,
        participant_classification_report,
        participant_confusion_matrix,
        participant_preds,
    )

### write selected features ###     
def write_selected_features(feature_lists, model_name, outdir='results/selected_features'):
    """
    Writes out the selected features for each fold of a model to CSV files.
    
    Args:
        feature_lists (List[List[str]]): Each element is the list of selected features for one fold.
        model_name (str): 'male', 'female', or 'mixed'
        outdir (str): Directory to write files to.
    """
    os.makedirs(outdir, exist_ok=True)
    for fold, features in enumerate(feature_lists, 1):
        with open(os.path.join(outdir, f'{model_name}_features_fold_{fold}.txt'), 'w') as f:
            for feat in features:
                f.write(f"{feat}\n")