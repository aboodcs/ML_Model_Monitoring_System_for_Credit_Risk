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

    def clean_and_engineer(self, df):
        df_clean = df.copy()
        df_clean = df_clean.drop_duplicates()
        
        # Drop identifier columns early
        df_clean = df_clean.drop(columns=["customer_id", "name"], errors="ignore")

        # Impute missing values before feature engineering
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        for col in df_clean.select_dtypes(include=["object"]).columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

        # 1 & 2. Engineered Ratios
        df_clean['loan_to_income_ratio'] = df_clean['credit_limit'] / (df_clean['net_yearly_income'] + 1)
        df_clean['debt_to_income_ratio'] = df_clean['yearly_debt_payments'] / (df_clean['net_yearly_income'] + 1)
        
        # 3. Credit utilization bin
        df_clean['credit_utilization_bin'] = pd.cut(
            df_clean['credit_limit_used(%)'], 
            bins=[-1, 30, 70, 100], 
            labels=['Low', 'Medium', 'High']
        )
        
        # 4. Risk score (Adapted to available dataset features)
        # Using credit_score, prev_defaults, and default_in_last_6months
        df_clean['risk_score'] = (df_clean['prev_defaults'] * 50) + (df_clean['default_in_last_6months'] * 100) + (1000 - df_clean['credit_score']) + (df_clean['credit_limit']/1000)
        
        # 5. Drop low-importance demographic features
        cols_to_drop = ["gender", "owns_car", "owns_house", "no_of_children", "migrant_worker", "total_family_members"]
        df_clean = df_clean.drop(columns=[col for col in cols_to_drop if col in df_clean.columns])

        # Outlier removal (skip target)
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
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
        logger.info(f"Data cleaned & engineered: {len(df)} rows → {len(df_clean)} rows")
        return df_clean

    def initiate_data_transformation(self):
        df = pd.read_csv(self.config.data_path)
        logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

        df_clean = self.clean_and_engineer(df)

        # One-hot encode categorical columns
        categorical_cols = df_clean.select_dtypes(include=["object", "category"]).columns.tolist()
        if categorical_cols:
            df_clean = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)
            logger.info(f"One-hot encoded categorical columns. New shape: {df_clean.shape}")

        # Separate features and target
        target_col = "credit_card_default"
        y = df_clean.pop(target_col)
        X = df_clean

        # 60/20/20 split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, random_state=42, stratify=y, test_size=0.2)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, random_state=42, stratify=y_temp, test_size=0.25)

        # Fit scaler ONLY on training data
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
