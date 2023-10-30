import sys
import os
from helpers import load_csv_data, create_csv_submission
from run_helpers import (
    load_useless_features_file,
    get_pearson_coefficients,
    get_spearman_coefficients,
    load_column_names_by_type,
    clean_data,
    clean_outliers,
    remove_small_variance_features,
    replace_nans_by_mean,
    z_score_normalization,
    remove_features_with_small_pearson_correlation,
    remove_features_with_small_spearman_correlation,
    run_model,
)
from implementations import (
    least_squares,
    reg_logistic_regression,
    ridge_regression,
    logistic_regression,
    mean_squared_error_sgd,
    mean_squared_error_gd,
)
import numpy as np

# Set directories path variables
ROOT_DIR = os.path.abspath(os.curdir)
DATA_DIR = os.path.join(ROOT_DIR, "data")
PREDICTIONS_DIR = os.path.join(ROOT_DIR, "predictions")
HELPER_FILES_DIR = os.path.join(ROOT_DIR, "helper_files")

# Set hyperparameters
models_to_run = ["least_squares"]
threshold = 0  # The threshold below which we return y = -1 and above which we return y = 1. Usually 0.
create_csv = False
models_parameters = {
    "least_squares": {},
    "reg_logistic_regression": {"lambda_": 0.0001, "max_iters": 10000, "gamma": 0.5},
    "ridge_regression": {"lambda_": 0.0001},
    "logistic_regression": {"max_iters": 10000, "gamma": 0.5},
    "mean_squared_error_sgd": {"max_iters": 10000, "gamma": 0.5},
    "mean_squared_error_gd": {"max_iters": 10000, "gamma": 0.5},
}
models_returned_values = {
    "least_squares": {"w": [], "loss": [], "y_pred": []},
    "reg_logistic_regression": {"w": [], "loss": [], "y_pred": []},
    "ridge_regression": {"w": [], "loss": [], "y_pred": []},
    "logistic_regression": {"w": [], "loss": [], "y_pred": []},
    "mean_squared_error_sgd": {"w": [], "loss": [], "y_pred": []},
    "mean_squared_error_gd": {"w": [], "loss": [], "y_pred": []},
}


# Load data from csv
(
    x_train_initial,
    x_test_initial,
    y_train_initial,
    train_ids_initial,
    test_ids_initial,
    column_names_initial,
) = load_csv_data(DATA_DIR)

# Create copies of the data
x_train, x_test, y_train, train_ids, test_ids, column_names = (
    x_train_initial.copy(),
    x_test_initial.copy(),
    y_train_initial.copy(),
    train_ids_initial.copy(),
    test_ids_initial.copy(),
    column_names_initial.copy(),
)

# Data has random values sometimes. Clean these.
(
    bools,
    seven_nines,
    seventyseven_ninetynine,
    specials,
    eight,
    eithgy_eight,
    fruits,
) = load_column_names_by_type(os.path.join(HELPER_FILES_DIR, "variables_by_values.csv"))
x_train = clean_data(
    x_train,
    column_names,
    bools,
    seven_nines,
    seventyseven_ninetynine,
    specials,
    eight,
    eithgy_eight,
    fruits,
)
x_test = clean_data(
    x_test,
    column_names,
    bools,
    seven_nines,
    seventyseven_ninetynine,
    specials,
    eight,
    eithgy_eight,
    fruits,
)

# Remove useless features from x_train and x_test (useless features are features regarding the identification of the patient)
useless_features_names = load_useless_features_file(
    os.path.join(HELPER_FILES_DIR, "useless_features_names.csv")
)
useless_columns_indices = np.where(np.in1d(column_names, useless_features_names))[0]
x_train = np.delete(x_train, useless_columns_indices, axis=1)
x_test = np.delete(x_test, useless_columns_indices, axis=1)
column_names = np.delete(column_names, useless_columns_indices)

# Clean outliers (defined as < 3rd percentile or > 97th percentile
x_train = clean_outliers(x_train)
x_test = clean_outliers(x_test)

# Remove small variance features
x_train, x_test = remove_small_variance_features(x_train, x_test)

# Replace nans by the mean of the feature
x_train = replace_nans_by_mean(x_train)
x_test = replace_nans_by_mean(x_test)

# Normalize the data
x_train = z_score_normalization(x_train)
x_test = z_score_normalization(x_test)

# Calculate Pearson and Spearman correlation between each feature and the target variable
# Remove the features where the correlation coefficient is close to zero
# (Defined as |coefficient| < 0.01)
x_train, x_test = remove_features_with_small_pearson_correlation(
    x_train, y_train, x_test
)
x_train, x_test = remove_features_with_small_spearman_correlation(
    x_train, y_train, x_test
)

# Run all models and return w, loss, y_pred
for model_name in models_to_run:
    (
        models_returned_values[model_name]["w"],
        models_returned_values[model_name]["loss"],
        models_returned_values[model_name]["y_pred"],
    ) = run_model(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        model_name=model_name,
        model_parameters=models_parameters[model_name],
        threshold=threshold,
    )

    # Check if we want to save csv file
    if create_csv:
        create_csv_submission(
            test_ids,
            models_returned_values[model_name]["y_pred"],
            os.path.join(PREDICTIONS_DIR, model_name + "_final.csv"),
        )
