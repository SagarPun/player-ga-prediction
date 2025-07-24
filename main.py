import pandas as pd
import os
from scripts.preprocess import preprocess_data
from scripts.model_keras import train_and_evaluate_keras
from scripts.shap_explainer import explain_model_with_shap, explain_single_player


def load_raw_data():
    path_2324 = os.path.join("data", "top5-players23-24.xlsx")
    path_2425 = os.path.join("data", "top5-players24-25.xlsx")

    df_2324 = pd.read_excel(path_2324)
    df_2425 = pd.read_excel(path_2425)

    print(f"Loaded 2023/24 data shape: {df_2324.shape}")
    print(f"Loaded 2024/25 data shape: {df_2425.shape}")

    return df_2324, df_2425

def main():
    df_2324, df_2425 = load_raw_data()
    X_train, y_train, X_test, y_test = preprocess_data(df_2324, df_2425)

    print(f"\nX_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    train_and_evaluate_keras(X_train, y_train, X_test, y_test,df_2425)
    explain_single_player(X_test, player_index=0)


if __name__ == "__main__":
    main()
