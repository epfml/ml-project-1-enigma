import csv
import os
import numpy as np
from helpers import create_csv_submission
from implementations import (
    least_squares,
    reg_logistic_regression,
    ridge_regression,
    logistic_regression,
    mean_squared_error_sgd,
    mean_squared_error_gd,
)


def load_useless_features_file(file_path):
    """
    Loads the list of useless features contained in the file at file_path
    Args:
        file_path: path to the file. File has to be single column.

    Returns:
        numpy array containing the list of useless features
    """
    useless_features_list = np.genfromtxt(
        file_path, delimiter=",", skip_header=1, dtype=str
    )

    return useless_features_list


def get_pearson_coefficients(X, y):
    """
    Calculates the pearson coefficients for each feature in X in relation to y.
    Args:
        X: All data, potentially multiple features
        y: Target data

    Returns:
    A list containing the Pearson coefficient of each feature
    """
    # Number of features
    num_features = X.shape[1]

    # Pearson Correlation Coefficients between each feature and the target variable
    pearson_coefficients = np.zeros(num_features)

    # Calculate Pearson Coefficient for each feature with the target
    for i in range(num_features):
        # Mean of feature and target
        mean_X = np.mean(X[:, i])
        mean_y = np.mean(y)

        # Standard deviation of feature and target
        std_X = np.std(X[:, i])
        std_y = np.std(y)

        # Covariance between feature and target
        covariance = np.mean((X[:, i] - mean_X) * (y - mean_y))

        # Pearson Correlation Coefficient between feature and target
        pearson_coefficients[i] = covariance / (std_X * std_y)

    return pearson_coefficients


def get_spearman_coefficients(X, y):
    """
    Calculates the Spearman coefficients for each feature in X in relation to y.
    Args:
        X: All data, potentially multiple features
        y: Target data

    Returns:
    A list containing the Spearman coefficient of each feature
    """

    def spearman_rank_correlation(x, y):
        """
        Calculate the Spearman rank-order correlation coefficient between two variables.

        :param x: array-like, the first variable
        :param y: array-like, the second variable
        :return: Spearman rank-order correlation coefficient
        """

        # Calculate the ranks of x and y
        rank_x = np.argsort(np.argsort(x))
        rank_y = np.argsort(np.argsort(y))

        # Calculate the covariance between the ranks of x and y
        covariance = np.mean((rank_x - rank_x.mean()) * (rank_y - rank_y.mean()))

        # Calculate the standard deviations of the rank variables
        std_rank_x = rank_x.std()
        std_rank_y = rank_y.std()

        # Calculate the Spearman correlation coefficient
        spearman_corr = covariance / (std_rank_x * std_rank_y)

        return spearman_corr

    # Calculate the Spearman correlation coefficients for each feature with the target variable
    spearman_coefficients = np.array(
        [spearman_rank_correlation(X[:, i], y) for i in range(X.shape[1])]
    )

    return spearman_coefficients


def load_column_names_by_type(file_path):
    """
    Loads the file containing the listed features that are of a specific type.
    Will then be used to clean data.
    Args:
        file_path: The path of the tile.

    Returns:
    bools, seven_nines, seventyseven_ninetynine, specials, eight, eithgy_eight, fruits.
    Each are np.array containing the names of the features that belong to their type.
    """
    all_names = np.genfromtxt(file_path, delimiter=",", skip_header=1, dtype=str)

    with open(file_path, "r", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        categories = np.array(next(reader))

    bools = all_names[:, np.where(categories == "bool_and_seven_nine_should_be_nan")]
    seven_nines = all_names[:, np.where(categories == "seven_nine_should_be_nan")]
    seventyseven_ninetynine = all_names[
        :, np.where(categories == "seventy_seven_and_ninety_nine_should_be_nan")
    ]
    specials = all_names[:, np.where(categories == "special")]
    eight = all_names[:, np.where(categories == "eight_should_be_zero")]
    eithgy_eight = all_names[:, np.where(categories == "eighty_eight_should_be_zero")]
    fruits = all_names[:, np.where(categories == "fruits_specials")]
    return (
        bools,
        seven_nines,
        seventyseven_ninetynine,
        specials,
        eight,
        eithgy_eight,
        fruits,
    )


def clean_data(
    data,
    column_names,
    bools,
    seven_nines,
    seventyseven_ninetynine,
    specials,
    eight,
    eithgy_eight,
    fruits,
):
    """
    Most of the data is not great. We use this function to apply a few sets of rules depending on the type of each column.
    Example: In most of the bools columns, a value of 1 means yes, a value of 2 means no, a value of 7 or 9 means
    the patient didn't answer the question. We change that to 1 means yes, -1 means no, 7 and 9 become nans as they should be
    Args:
        data: The data containing all features
        column_names: The name of each column. In the same order as the column of data
        bools: List of features of type bool
        seven_nines: List of features of type seven_nine
        seventyseven_ninetynine: List of features of type seventyseven_ninetynine
        specials: List of features of type specials
        eight: List of features of type eight
        eithgy_eight: List of features of type eithgy_eight
        fruits: List of features of type fruits

    Returns:
    The cleaned up data
    """
    copied_data = np.copy(data)

    bools_columns = np.where(np.in1d(column_names, bools))[0]
    seven_nines_columns = np.where(np.in1d(column_names, seven_nines))[0]
    seventyseven_ninetynine_columns = np.where(
        np.in1d(column_names, seventyseven_ninetynine)
    )[0]
    specials_columns = np.where(np.in1d(column_names, specials))[0]
    eight_columns = np.where(np.in1d(column_names, eight))[0]
    eithgy_eight_columns = np.where(np.in1d(column_names, eithgy_eight))[0]
    fruits_columns = np.where(np.in1d(column_names, fruits))[0]

    # Rule for bools: 7 and 9 become np.nan, 2 and 0 become -1
    for index in bools_columns:
        copied_data = replace_values(
            copied_data, column_index=index, value=7, new_value=np.nan
        )
        copied_data = replace_values(
            copied_data, column_index=index, value=9, new_value=np.nan
        )
        copied_data = replace_values(
            copied_data, column_index=index, value=2, new_value=-1
        )
        copied_data = replace_values(
            copied_data, column_index=index, value=0, new_value=-1
        )

    # Rule for seven_nines: 7 and 9 become np.nan
    for index in seven_nines_columns:
        copied_data = replace_values(
            copied_data, column_index=index, value=7, new_value=np.nan
        )
        copied_data = replace_values(
            copied_data, column_index=index, value=9, new_value=np.nan
        )

    # Rule for seventyseven_ninetynine: 77 and 99 become np.nan
    for index in seventyseven_ninetynine_columns:
        copied_data = replace_values(
            copied_data, column_index=index, value=77, new_value=np.nan
        )
        copied_data = replace_values(
            copied_data, column_index=index, value=99, new_value=np.nan
        )

    # Rule for eight: 8 becomes 0
    for index in eight_columns:
        copied_data = replace_values(
            copied_data, column_index=index, value=8, new_value=0
        )

    # Rule for eighty_eight: 88 becomes 0
    for index in eithgy_eight_columns:
        copied_data = replace_values(
            copied_data, column_index=index, value=8, new_value=0
        )

    # Rule for special and fruits: CURRENTLY NOTHING

    return copied_data


def replace_values(data, column_index, value, new_value):
    """
    Replaces all elements that are value by new_value in the column at column_index of data.
    Args:
        data: The data containing all features
        column_index: column index of feature to modify
        value: old value
        new_value: changed value

    Returns:
    The data with the modified column
    """
    feature = data[:, column_index]
    feature[feature == value] = new_value
    data[:, column_index] = feature
    return data


def clean_outliers(data):
    """
    Cleans the dataset from outliers.
    The first and last 3 percentile outliers are replaced by the median without outliers.
    Args:
        data: The data

    Returns:
    The cleaned data
    """
    cleaned_data = np.copy(data)

    for i in range(data.shape[1]):  # Iterate over features/columns
        feature = data[:, i]

        # Compute the first and last 3% and IQR
        Q1 = np.nanpercentile(feature, 3)
        Q3 = np.nanpercentile(feature, 97)
        IQR = Q3 - Q1

        # Identify the outliers
        outlier_mask = (feature < (Q1 - 1.5 * IQR)) | (feature > (Q3 + 1.5 * IQR))

        # Compute the median of the data without outliers
        median_without_outliers = np.nanmedian(feature[~outlier_mask])

        # Replace outliers with this median
        cleaned_data[outlier_mask, i] = median_without_outliers

    return cleaned_data


def remove_small_variance_features(data_train, data_test):
    """
    Removed features with a variance over mean ratio smaller than 0.01
    Args:
        data_train: the training set (X_train)
        data_test: the testing set (X_test

    Returns:
    The cleaned up X_train and X_test
    """
    cleaned_data_train = np.copy(data_train)
    cleaned_data_test = np.copy(data_test)
    # Calculate the variance for each feature
    variances_over_means = np.abs(
        np.nanvar(cleaned_data_train, axis=0) / np.nanmean(cleaned_data_train, axis=0)
    )

    # Set your threshold for variance (e.g., 0.01)
    threshold = 0.01

    # Find feature indices that meet the threshold
    features_to_keep = variances_over_means >= threshold

    # Keep only the features with variance above the threshold
    data_reduced_train = cleaned_data_train[:, features_to_keep]
    data_reduced_test = cleaned_data_test[:, features_to_keep]

    return data_reduced_train, data_reduced_test


def replace_nans_by_mean(data):
    """
    Replaces all nans in data by the mean of their column.
    Args:
        data: The data

    Returns:
    The cleaned up data
    """
    copied_data = np.copy(data)
    means = np.nanmean(copied_data, axis=0)

    # Replace nan values with the computed means for each feature
    for i in range(copied_data.shape[1]):
        copied_data[np.isnan(copied_data[:, i]), i] = means[i]

    return copied_data


def z_score_normalization(data):
    """
    Normalizes every column of data.
    Args:
        data: The data

    Returns:
    The normalized data
    """
    mean_vals = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    return (data - mean_vals) / std_dev


def remove_features_with_small_pearson_correlation(data_train, target_train, data_test):
    """
    Removes all features with a |Pearson correlation to the target variable| smaller than 0.01 (in absolute value).
    Args:
        data_train: The training set (X_train)
        target_train: The training target variables (y_train)
        data_test: The testing set (X_test)

    Returns:
    The cleaned up X_train and X_test
    """
    cleaned_data_train = np.copy(data_train)
    cleaned_data_test = np.copy(data_test)

    pearson_coeffs = get_pearson_coefficients(data_train, target_train)

    threshold = 0.01

    # Find feature indices that meet the threshold
    features_to_keep = np.abs(pearson_coeffs) >= threshold

    # Keep only the features with variance above the threshold
    data_reduced_train = cleaned_data_train[:, features_to_keep]
    data_reduced_test = cleaned_data_test[:, features_to_keep]

    return data_reduced_train, data_reduced_test


def remove_features_with_small_spearman_correlation(
    data_train, target_train, data_test
):
    """
    Removes all features with a |Spearman correlation to the target variable| smaller than 0.01 (in absolute value).
    Args:
        data_train: The training set (X_train)
        target_train: The training target variables (y_train)
        data_test: The testing set (X_test)

    Returns:
    The cleaned up X_train and X_test
    """
    cleaned_data_train = np.copy(data_train)
    cleaned_data_test = np.copy(data_test)

    spearman_coeffs = get_spearman_coefficients(data_train, target_train)

    threshold = 0.01

    # Find feature indices that meet the threshold
    features_to_keep = np.abs(spearman_coeffs) >= threshold

    # Keep only the features with variance above the threshold
    data_reduced_train = cleaned_data_train[:, features_to_keep]
    data_reduced_test = cleaned_data_test[:, features_to_keep]

    return data_reduced_train, data_reduced_test


def run_model(
    x_train,
    y_train,
    x_test,
    model_name,
    model_parameters,
    threshold,
):
    """
    Runs the given model and outputs the weights, loss and predictions of this model
    Args:
        x_train: The training set
        y_train: The training target variables
        x_test: The testing set
        model_name: The name of the model
        model_parameters: The parameters of the model in the form of a dict
        threshold: The threshold below which a value is classified as -1 and above which it is classified as 1 (usually 0)

    Returns:
        w: The last weight vector of the method
        loss: The corresponding loss value (cost function)
        y_pred: The predictions of the model over x_test
    """
    if model_name == "least_squares":
        w, loss = least_squares(y=y_train, tx=x_train)
    elif model_name == "reg_logistic_regression":
        w, loss = reg_logistic_regression(
            y=y_train,
            tx=x_train,
            lambda_=model_parameters["lambda_"],
            initial_w=np.zeros(x_train.shape[1]),
            max_iters=model_parameters["max_iters"],
            gamma=model_parameters["gamma"],
        )
    elif model_name == "ridge_regression":
        w, loss = ridge_regression(
            y=y_train, tx=x_train, lambda_=model_parameters["lambda_"]
        )
    elif model_name == "logistic_regression":
        w, loss = logistic_regression(
            y=y_train,
            tx=x_train,
            initial_w=np.zeros(x_train.shape[1]),
            max_iters=model_parameters["max_iters"],
            gamma=model_parameters["gamma"],
        )
    elif model_name == "mean_squared_error_sgd":
        w, loss = mean_squared_error_sgd(
            y=y_train,
            tx=x_train,
            initial_w=np.zeros(x_train.shape[1]),
            max_iters=model_parameters["max_iters"],
            gamma=model_parameters["gamma"],
        )
    elif model_name == "mean_squared_error_gd":
        w, loss = mean_squared_error_gd(
            y=y_train,
            tx=x_train,
            initial_w=np.zeros(x_train.shape[1]),
            max_iters=model_parameters["max_iters"],
            gamma=model_parameters["gamma"],
        )
    else:
        w, loss = [], []

    y_pred = x_test @ w
    y_pred[y_pred >= threshold] = 1
    y_pred[y_pred < threshold] = -1

    return w, loss, y_pred
