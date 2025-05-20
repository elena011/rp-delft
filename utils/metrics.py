import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error


def evaluate_age_prediction(y_true, y_pred, model_name="Model", cpg_count=None):
    """
    Evaluates prediction metrics for age estimation models.

    Parameters:
    - y_true (array-like): Ground truth ages.
    - y_pred (array-like): Predicted ages.
    - model_name (str): Name of the model being evaluated.
    - cpg_count (int): Number of CpGs used in the model.

    Returns:
    - pd.DataFrame: Metrics including MAE, MSE, MAD, Pearson R, and Median Error.
    """

    y_true = np.array(y_true).astype(float)
    y_pred = np.array(y_pred).astype(float)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mad = median_absolute_error(y_true, y_pred)
    r, _ = pearsonr(y_true, y_pred)
    median_error = np.median(y_true - y_pred)

    results_df = pd.DataFrame({
        'Model': [model_name],
        'CpGs': [cpg_count],
        'MAD': [mad],
        'MAE': [mae],
        'MSE': [mse],
        'Pearson R': [r],
        'Median Error': [median_error]
    })

    return results_df
