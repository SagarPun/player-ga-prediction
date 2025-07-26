## Setup

To set up the project environment, we recommend using [Conda](https://docs.conda.io/en/latest/miniconda.html).

1.  **Create and activate the Conda environment:**
    ```bash
    conda env create -f environment.yml
    conda activate dlplayers
    ```
2.  **Ensure data files are in place:**
    Make sure your `top5-players23-24.xlsx` and `top5-players24-25.xlsx` files are located in the `data/` directory.

## Usage

You can configure and run the entire pipeline through `main.py`.

1.  **Configure `config.py`:**
    Open `config.py` to adjust parameters. The `project_config["selected_model"]` flag allows you to choose which model pipeline (`"keras"` or `"sklearn"`) to execute. Both models will generate their respective predictions, SHAP plots, and diagnostic visualizations.

    ```python
    # config.py
    project_config = {
        "selected_model": "keras" # Change to "sklearn" to run the Scikit-learn model pipeline
    }
    ```

2.  **Run the main script:**
    Execute the `main.py` script from your project's root directory:
    ```bash
    python main.py
    ```
    The script will print status messages and MAE results to the console. All generated plots and CSVs will be saved in the `output/` directory.

## Output Files

* `output/predictions_keras.csv`: Predictions from the Keras model.
* `output/predictions_sklearn.csv`: Predictions from the Scikit-learn model.
* `output/shap_summary_keras.png`: SHAP summary plot for the Keras model.
* `output/shap_summary_sklearn.png`: SHAP summary plot for the Scikit-learn model.
* `output/actual_vs_predicted_keras.png`: Actual vs. Predicted plot for Keras.
* `output/actual_vs_predicted_sklearn.png`: Actual vs. Predicted plot for Scikit-learn.
* `output/residuals_distribution_keras.png`: Residuals distribution for Keras.
* `output/residuals_distribution_sklearn.png`: Residuals distribution for Scikit-learn.
* `output/residuals_vs_predicted_keras.png`: Residuals vs. Predicted for Keras.
* `output/residuals_vs_predicted_sklearn.png`: Residuals vs. Predicted for Scikit-learn.
* `output/feature_correlation_heatmap.png`: Correlation heatmap of features.

## Key Learnings and Insights

Based on the implemented models and analyses:

* **Model Performance:** For this dataset, the Scikit-learn MLP Regressor (MAE: 0.09) significantly outperformed the custom Keras Deep Learning model (MAE: 0.25). This highlights that model complexity doesn't always guarantee superior raw predictive performance, and robust hyperparameter tuning for simpler models can be highly effective.
* **Interpretability:** SHAP analysis for both models generally confirmed the intui~~~~~~~~tive importance of offensive statistics (Goals per 90, Expected Goals, Assists per 90) in predicting G+A. While both models agreed on top features, the Scikit-learn model exhibited tighter SHAP value distributions, possibly indicating a more consistent and precise learned relationship.
* **Fairness:** Initial fairness audits showed that the Scikit-learn model maintained a consistently lower MAE across different player positions and age groups compared to the Keras model, suggesting more equitable predictions in terms of error magnitude.
* **Development Process:** The project underscored the iterative nature of deep learning development, where initial assumptions and approaches often evolve in response to encountered challenges (e.g., Keras model saving, SHAP computation time).
