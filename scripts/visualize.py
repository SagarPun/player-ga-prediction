import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_actual_vs_predicted(y_true, y_pred, save_path):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel("Actual G+A")
    plt.ylabel("Predicted G+A")
    plt.title("Actual vs. Predicted G+A")
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Actual vs. Predicted plot saved to {save_path}")

def plot_residuals_distribution(y_true, y_pred, save_path):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, bins=30)
    plt.xlabel("Residuals (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Prediction Residuals")
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Residuals distribution plot saved to {save_path}")

def plot_residuals_vs_predicted(y_true, y_pred, save_path):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel("Predicted G+A")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Residuals vs. Predicted G+A")
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Residuals vs. Predicted plot saved to {save_path}")

def plot_feature_correlation_heatmap(X_data, y_data, save_path):
    df_combined = X_data.copy()
    df_combined['Target'] = y_data.copy()

    plt.figure(figsize=(12, 10))
    sns.heatmap(df_combined.corr(), annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Feature Correlation Heatmap saved to {save_path}")