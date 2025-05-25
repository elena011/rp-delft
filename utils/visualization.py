import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_predicted_vs_true(true_age, predicted_age, colour, model_name):
    """
    Plots predicted age vs. true chronological age.

    Parameters:
    - true_age (array-like): True chronological ages.
    - predicted_age (array-like): Predicted ages from a model.
    """
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=true_age, y=predicted_age, alpha=0.6, color=colour)

    # Plot the diagonal reference line
    min_age = min(min(true_age), min(predicted_age))
    max_age = max(max(true_age), max(predicted_age))
    plt.plot([min_age, max_age], [min_age, max_age], linestyle='--', color='red', label='Perfect Prediction')

    plt.xlabel("True Age", fontsize=16)
    plt.ylabel("Predicted Age", fontsize=16)
    plt.title("Predicted vs. True Age -- " + model_name, fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()


def plot_feature_importance(model, feature_names, top_n=20, title='Feature Importance', plot=True):
    """
    Get or plot feature importance for models with .coef_ or .feature_importances_ attribute.

    Parameters:
        model: Trained model (e.g., LinearRegression, RandomForest)
        feature_names: List or array of feature names
        top_n: Number of top features to return
        title: Title of the plot (if plotting)
        plot: If True, plot the feature importance bar chart. If False, just return the DataFrame.

    Returns:
        DataFrame of top features and their importance.
    """
    # Determine importance type
    if hasattr(model, 'coef_'):
        importance_values = model.coef_
    elif hasattr(model, 'feature_importances_'):
        importance_values = model.feature_importances_
    else:
        raise ValueError("Model must have either .coef_ or .feature_importances_ attribute.")

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_values,
        'AbsImportance': np.abs(importance_values)
    }).sort_values(by='AbsImportance', ascending=False).head(top_n)

    if plot:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df,
                    palette=['#1d5eaa' if val > 0 else '#b22222' for val in importance_df['Importance']])
        plt.title(title)
        plt.xlabel('Coefficient')
        plt.ylabel('CpG Site')
        plt.tight_layout()
        plt.show()

    return importance_df

def plot_residuals(true_age, predicted_age):
    """
    Plots residuals (predicted - true) against true age.
    """

    residuals = [p - t for p, t in zip(predicted_age, true_age)]

    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=true_age, y=residuals, alpha=0.6)
    plt.axhline(0, linestyle='--', color='red')
    plt.xlabel("True Age")
    plt.ylabel("Residual (Predicted - True)")
    plt.title("Residual Plot")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_mae_rmse_boxplot(results_df):
    """
    Plots boxplots for MAE and RMSE grouped by model.

    Parameters:
    - results_df (pd.DataFrame): Must contain columns ['model', 'MAE', 'RMSE']
    """

    melted = results_df.melt(id_vars='model', value_vars=['MAE', 'RMSE'], var_name='Metric', value_name='Value')

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=melted, x='model', y='Value', hue='Metric')
    plt.title("Boxplot of MAE and RMSE per Model")
    plt.xlabel("Model")
    plt.ylabel("Error")
    plt.grid(True)
    plt.tight_layout()
    plt.legend(title='Metric')
    plt.show()


def plot_r2_barplot(r2_scores_df):
    """
    Plots a bar chart of R² scores across models and datasets.

    Parameters:
    - r2_scores_df (pd.DataFrame): Must contain ['dataset', 'model', 'R2']
    """

    plt.figure(figsize=(8, 5))
    sns.barplot(data=r2_scores_df, x='dataset', y='R2', hue='model')
    plt.title("R² Scores Across Datasets")
    plt.ylabel("R²")
    plt.xlabel("Dataset")
    plt.ylim(0, 1)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.legend(title='Model')
    plt.show()
