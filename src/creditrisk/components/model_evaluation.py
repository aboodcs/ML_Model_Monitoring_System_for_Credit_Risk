import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from urllib.parse import urlparse
from src.creditrisk.entity.config_entity import ModelEvaluationConfig
from src.creditrisk.utils import read_yaml, create_directories ,save_json
from pathlib import Path
import os

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/aboodcs/ML_Model_Monitoring_System_for_Credit_Risk.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "aboodcs"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "6101bda6b23aa9ef440664cb2f0aac4cb03e1bcf"

class ModelEvaluation:
    def __init__(self,config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual , pred):
        rmse = np.sqrt(mean_squared_error(actual , pred))
        mae = mean_absolute_error(actual , pred)
        r2 = r2_score(actual , pred)
        return rmse, mae, r2
    
    def log_into_mlflow(self):
        test_x = pd.read_csv(self.config.test_data_path)
        test_y = pd.read_csv(str(self.config.test_data_path).replace("X_test.csv", "y_test.csv"))
        test_y = test_y.values.ravel()
        model = joblib.load(self.config.model_path)

        mlflow.set_registry_uri(self.config.mlflow_uri)
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            predicted_qualities = model.predict(test_x)
            (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)
            mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})

            #saving metrices as local
            scores = {
                "rmse": rmse,
                "mae": mae,
                "r2": r2}
            save_json(path=Path(self.config.metric_file_path), data=scores)

            mlflow.log_params(self.config.all_params)
            
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, name="model", registered_model_name="ElasticNetWineModel")
            else:
                mlflow.sklearn.log_model(model, name="model")