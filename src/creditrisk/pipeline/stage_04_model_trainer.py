from src.creditrisk.config.configuration import ConfigurationManager
from src.creditrisk.components.model_trainer import ModelTrainer
from src.creditrisk.utils import logger

STAGE_NAME = "Model Trainer Stage"

class ModelTrainerPipeline:
    def __init__(self):
        pass
    def initiate_model_trainer(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.train()