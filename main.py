import sys
import os

_project_root = os.path.dirname(os.path.abspath(__file__))
_venv_site_packages = os.path.join(_project_root, "env", "lib", "python3.12", "site-packages")
if _venv_site_packages not in sys.path:
    sys.path.insert(0, _venv_site_packages)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
# ─────────────────────────────────────────────────────────────────────────────

from src.creditrisk.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.creditrisk.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from src.creditrisk.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from src.creditrisk.pipeline.stage_04_model_trainer import ModelTrainerPipeline
from src.creditrisk.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline
from src.creditrisk.utils import logger


STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.initiate_data_ingestion()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise


STAGE_NAME = "Data Validation Stage"
try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    data_validation = DataValidationTrainingPipeline()
    data_validation.initiate_data_validation()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise


STAGE_NAME = "Data Transformation Stage"
try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.initiate_data_transformation()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise


STAGE_NAME = "Model Trainer Stage"
try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    model_trainer = ModelTrainerPipeline()
    model_trainer.initiate_model_trainer()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise


STAGE_NAME = "Model Evaluation Stage"
try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    model_evaluation = ModelEvaluationPipeline()
    model_evaluation.initiate_model_evaluation()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
