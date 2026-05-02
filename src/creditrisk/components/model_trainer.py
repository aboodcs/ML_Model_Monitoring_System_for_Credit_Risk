import os
from src.creditrisk.utils import logger
from src.creditrisk.entity.config_entity import ModelTrainerConfig

import pandas as pd
from sklearn.linear_model import ElasticNet
import joblib

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_x = pd.read_csv(self.config.train_data_path)
        test_x  = pd.read_csv(self.config.test_data_path)
        train_y = pd.read_csv(str(self.config.train_data_path).replace("X_train.csv", "y_train.csv"))
        test_y  = pd.read_csv(str(self.config.test_data_path).replace("X_test.csv", "y_test.csv"))

        lr = ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42)
        lr.fit(train_x, train_y.values.ravel())
        joblib.dump(lr, str(self.config.model_path))
        logger.info(f"Model trained and saved at: {self.config.model_path}")