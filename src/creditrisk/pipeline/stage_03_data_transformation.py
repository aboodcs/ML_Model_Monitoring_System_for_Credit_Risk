from src.creditrisk.config.configuration import ConfigurationManager
from src.creditrisk.components.data_transformation import DataTransformation
from src.creditrisk.utils import logger
import os


STAGE_NAME = "Data Transformation Stage"


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_transformation(self):
        try:
            with open(os.path.join("artifacts", "data_validation", "status.txt"), "r") as f:
                status = f.read().split(" ")[-1]

            if status == "Passed":
                config = ConfigurationManager()
                data_transformation_config = config.get_data_transformation_config()
                data_transformation = DataTransformation(config=data_transformation_config)
                data_transformation.initiate_data_transformation()
            else:
                raise Exception("Data validation failed. Aborting transformation.")

        except Exception as e:
            logger.exception(e)
            raise e


if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.initiate_data_transformation()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
