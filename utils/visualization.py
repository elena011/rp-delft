import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

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


def plot_predicted_vs_true_by_tissue(test_info, true_ages, predicted_ages, title="Predicted vs True Age by Tissue Type", cols=4):
    """
    Plot predicted vs. true age for each tissue type with R² and MAE, and no overlapping titles or empty subplots.
    """
    # Combine data
    plot_df = test_info.copy()
    plot_df['true_age'] = true_ages.values
    plot_df['predicted_age'] = predicted_ages

    sns.set(style="whitegrid")
    tissues = plot_df['tissue_type'].unique()
    n_tissues = len(tissues)
    rows = int(np.ceil(n_tissues / cols))

    # Use constrained_layout to avoid overlaps
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), constrained_layout=True)
    axes = axes.flatten()

    for idx, tissue in enumerate(tissues):
        ax = axes[idx]
        tissue_data = plot_df[plot_df['tissue_type'] == tissue]

        r2 = r2_score(tissue_data['true_age'], tissue_data['predicted_age'])
        mae = mean_absolute_error(tissue_data['true_age'], tissue_data['predicted_age'])

        sns.scatterplot(x='true_age', y='predicted_age', data=tissue_data, ax=ax)
        ax.plot(tissue_data['true_age'], tissue_data['true_age'], color='red', linestyle='--')

        ax.set_title(f"{tissue}\nR²={r2:.2f}, MAE={mae:.1f}")
        ax.set_xlabel('True Age')
        ax.set_ylabel('Predicted Age')

    # Remove unused axes
    for j in range(n_tissues, len(axes)):
        fig.delaxes(axes[j])

    # Set global title
    fig.suptitle(title, fontsize=20)
    plt.show()
