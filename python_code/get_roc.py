import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer

# Define paths
data_path = os.path.join("data", "speech_data")
features_path = os.path.join("data", "feature_lists")
model_base_path = os.path.join("data", "models")

# Load test sets
test_sets = {
    "male": pd.read_csv(os.path.join(data_path, "data_q_male_test.csv")),
    "female": pd.read_csv(os.path.join(data_path, "data_q_female_test.csv")),
    "mixed": pd.read_csv(os.path.join(data_path, "data_q_mixed_test.csv"))
}

# Load stable features
n_folds = 3

stable_df_male = pd.read_csv(os.path.join(features_path, "features_data_q_male_STABLE.csv"))
stable_features_male = stable_df_male["features"].str.strip("`").tolist()
features_male = [stable_features_male for _ in range(n_folds)]

stable_df_female = pd.read_csv(os.path.join(features_path, "features_data_q_female_STABLE.csv"))
stable_features_female = stable_df_female["features"].str.strip("`").tolist()
features_female = [stable_features_female for _ in range(n_folds)]

stable_df_mixed = pd.read_csv(os.path.join(features_path, "features_data_q_mixed_STABLE.csv"))
stable_features_mixed = stable_df_mixed["features"].str.strip("`").tolist()
features_mixed = [stable_features_mixed for _ in range(n_folds)]

# put them in the same structure used later in the script
feature_sets = {
    "male": features_male,
    "female": features_female,
    "mixed": features_mixed
}


# Ensemble ROC AUC Calculation OBSERVATION LEVEL
model_groups = ["male", "female", "mixed"]

for model_group in model_groups:
    print(f"### Evaluating ensemble model (obs level): {model_group} ###")

    model_dir = os.path.join(model_base_path, f"model_{model_group}")
    model_files = [os.path.join(model_dir, f"rbf_{i+1}.pkl") for i in range(3)]

    # Load models
    ensemble_models = []
    for fpath in model_files:
        if not os.path.exists(fpath):
            print(f"  Missing model file: {fpath}")
            break
        ensemble_models.append(joblib.load(fpath))

    if len(ensemble_models) != 3:
        print("  Skipping: incomplete ensemble.\n")
        continue

    for test_group, test_df in test_sets.items():
        print(f"  -> Testing on: {test_group}")
        
        # Prepare test data
        selected_features = feature_sets[model_group]
        y = test_df["Diagnosis"]
        lb = LabelBinarizer()
        y_bin = lb.fit_transform(y).ravel()

        decision_values = []

        for i, model in enumerate(ensemble_models):
            feature_list = feature_sets[model_group][i]
            
            # Filter and reindex test_df with only features the model expects
            expected_features = model.feature_names_in_
            testX = test_df.copy()

            # Ensure we keep only the expected features, and in the correct order
            missing_feats = [f for f in expected_features if f not in testX.columns]
            if missing_feats:
                raise ValueError(f"Missing expected features: {missing_feats}")

            testX = testX[expected_features]

            assert list(testX.columns) == list(expected_features), f"Feature mismatch for fold {i+1}"

            # Get decision function scores
            decision_values.append(model.decision_function(testX))

        decision_values = np.vstack(decision_values)
        ensemble_score = np.mean(decision_values, axis=0)

        # Compute ROC AUC
        auc = roc_auc_score(y_bin, ensemble_score)
        print(f"     ROC AUC: {auc:.3f}")
    print()



# Ensemble ROC AUC Calculation PARTICIPANT LEVEL
for model_group in model_groups:
    print(f"### Evaluating ensemble model (part level): {model_group} ###")

    model_dir = os.path.join(model_base_path, f"model_{model_group}")
    model_files = [os.path.join(model_dir, f"rbf_{i+1}.pkl") for i in range(n_folds)]
    ensemble_models = [joblib.load(f) for f in model_files if os.path.exists(f)]

    if len(ensemble_models) != n_folds:
        print("  Skipping: incomplete ensemble.\n")
        continue

    for test_group, test_df in test_sets.items():
        print(f"  -> Testing on: {test_group}")

        y_true = []
        y_scores = []

        # Loop over each participant
        for pid, group in test_df.groupby("ID"):
            participant_scores = []

            for i, model in enumerate(ensemble_models):
                feature_list = feature_sets[model_group][i]
                if not all(f in group.columns for f in feature_list):
                    print(f"Missing features for model {i+1}, skipping participant {pid}")
                    continue

                try:
                    # Get expected features from the model (trained with DataFrame input)
                    expected_features = model.feature_names_in_
                    X_part_aligned = group.reindex(columns=expected_features, fill_value=0.0)

                    scores = model.decision_function(X_part_aligned)
                except Exception as e:
                    print(f"Error for participant {pid}, model {i+1}: {e}")
                    continue

                participant_scores.append(scores)

            if participant_scores:
                avg_score = np.mean(np.vstack(participant_scores), axis=(0, 1))
                diagnosis = group["Diagnosis"].iloc[0]
                y_true.append(diagnosis)
                y_scores.append(avg_score)

        if len(set(y_true)) < 2:
            print("  Not enough class variation to compute ROC AUC.")
            continue

        lb = LabelBinarizer()
        y_bin = lb.fit_transform(y_true).ravel()
        auc = roc_auc_score(y_bin, y_scores)
        print(f"     Participant-level ROC AUC: {auc:.3f}")
    print()