import os
import pandas as pd
from lightgbm import LGBMClassifier
import joblib
import optuna
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from src.creditrisk.utils import logger
from src.creditrisk.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_x = pd.read_csv(self.config.train_data_path)
        train_y = pd.read_csv(str(self.config.train_data_path).replace('X_train.csv', 'y_train.csv'))
        train_y_vals = train_y.values.ravel()

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 80),
                'class_weight': 'balanced',
                'random_state': 42,
                'verbose': -1
            }
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            f1_scores = []
            for train_idx, val_idx in cv.split(train_x, train_y_vals):
                X_tr, X_val = train_x.iloc[train_idx], train_x.iloc[val_idx]
                y_tr, y_val = train_y_vals[train_idx], train_y_vals[val_idx]
                model = LGBMClassifier(**params)
                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)
                f1_scores.append(f1_score(y_val, preds))
            return sum(f1_scores) / len(f1_scores)

        logger.info('Starting Optuna optimization for LightGBM (100 trials)...')
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)

        best_params = study.best_params
        best_params['class_weight'] = 'balanced'
        best_params['random_state'] = 42
        best_params['verbose'] = -1
        logger.info(f'Best params found by Optuna: {best_params}')

        lgbm = LGBMClassifier(**best_params)
        lgbm.fit(train_x, train_y_vals)

        joblib.dump(lgbm, str(self.config.model_path))
        logger.info(f'Model trained and saved at: {self.config.model_path}')
