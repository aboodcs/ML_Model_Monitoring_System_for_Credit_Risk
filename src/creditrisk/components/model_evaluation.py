import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_curve
import joblib
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
from src.creditrisk.entity.config_entity import ModelEvaluationConfig
from src.creditrisk.utils import save_json, logger
from pathlib import Path

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/aboodcs/ML_Model_Monitoring_System_for_Credit_Risk.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "aboodcs"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "6101bda6b23aa9ef440664cb2f0aac4cb03e1bcf"


class ModelEvaluation:
    def __init__(self,config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual , pred):
        accuracy = accuracy_score(actual , pred)
        f1 = f1_score(actual , pred)
        precision = precision_score(actual , pred)
        recall = recall_score(actual , pred)
        return accuracy, f1, precision, recall
    
    def log_into_mlflow(self):
        test_x = pd.read_csv(self.config.test_data_path)
        test_y = pd.read_csv(str(self.config.test_data_path).replace("X_test.csv", "y_test.csv"))
        test_y = test_y.values.ravel()
        model = joblib.load(self.config.model_path)

        mlflow.set_registry_uri(self.config.mlflow_uri)
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            predicted_classes = model.predict(test_x)
            (accuracy, f1, precision, recall) = self.eval_metrics(test_y, predicted_classes)
            mlflow.log_metrics({"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall})

            #saving metrices as local
            scores = {
                "accuracy": accuracy,
                "f1": f1,
                "precision": precision,
                "recall": recall}
            save_json(path=Path(self.config.metric_file_path), data=scores)

            mlflow.log_params(self.config.all_params)
            
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)

            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, name="model", registered_model_name="ElasticNetWineModel")
            else:
                mlflow.sklearn.log_model(model, name="model")