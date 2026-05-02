import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.creditrisk.entity.config_entity import DataTransformationConfig
from src.creditrisk.utils import logger


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def clean_data(self, df):
        df_clean = df.copy()
        df_clean = df_clean.drop_duplicates()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        # Remove target from outlier removal
        target = "credit_card_default"
        if target in numeric_cols:
            numeric_cols.remove(target)

        outliers_mask = pd.Series([False] * len(df_clean), index=df_clean.index)
        for col in numeric_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            col_outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            outliers_mask = outliers_mask | col_outliers.values

        df_clean = df_clean[~outliers_mask]
        logger.info(f"Data cleaned: {len(df)} rows → {len(df_clean)} rows after outlier removal")
        return df_clean

    def initiate_data_transformation(self):
        df = pd.read_csv(self.config.data_path)
        logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

        # Drop non-feature identifier columns
        df = df.drop(columns=["customer_id", "name"], errors="ignore")

        # Impute missing values before cleaning
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(df[col].median())
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].fillna(df[col].mode()[0])

        df_clean = self.clean_data(df)

        # One-hot encode categorical columns
        categorical_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()
        if categorical_cols:
            df_clean = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)
            logger.info(f"One-hot encoded categorical columns. New shape: {df_clean.shape}")

        # Separate features and target
        target_col = "credit_card_default"
        y = df_clean.pop(target_col)
        X = df_clean

        # 60/20/20 split — test=20%, then val=25% of 80% = 20%
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, random_state=42, stratify=y, test_size=0.2)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, random_state=42, stratify=y_temp, test_size=0.25)

        # Fit scaler ONLY on training data to prevent data leakage
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
        X_val   = pd.DataFrame(scaler.transform(X_val),       columns=X.columns)
        X_test  = pd.DataFrame(scaler.transform(X_test),      columns=X.columns)

        logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        # Save splits
        X_train.to_csv(os.path.join(self.config.root_dir, "X_train.csv"), index=False)
        X_val.to_csv(  os.path.join(self.config.root_dir, "X_val.csv"),   index=False)
        X_test.to_csv( os.path.join(self.config.root_dir, "X_test.csv"),  index=False)
        y_train.to_csv(os.path.join(self.config.root_dir, "y_train.csv"), index=False)
        y_val.to_csv(  os.path.join(self.config.root_dir, "y_val.csv"),   index=False)
        y_test.to_csv( os.path.join(self.config.root_dir, "y_test.csv"),  index=False)

        # Save scaler
        with open(self.config.preprocessor_path, "wb") as f:
            pickle.dump(scaler, f)

        logger.info("Data transformation complete. Preprocessor and splits saved.")
