import os
import pandas as pd
from src.creditrisk.entity.config_entity import DataValidationConfig
from src.creditrisk.utils import logger


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        """
        Validate all columns in the dataset against the schema.
        Drops non-feature columns (customer_id, name) before checking.
        """
        try:
            validation_status = True

            data_file = os.path.join(self.config.unzip_data_dir, "train.csv")
            df = pd.read_csv(data_file)

            # Drop identifier columns not in schema
            df = df.drop(columns=["customer_id", "name"], errors="ignore")

            # Get expected columns from schema (features + target)
            all_schema_columns = list(self.config.all_schema.keys())

            df_columns = df.columns.tolist()

            for col in df_columns:
                if col not in all_schema_columns:
                    validation_status = False
                    logger.info(f"Column '{col}' not found in schema")

            if validation_status:
                logger.info("All columns validated successfully against schema")

            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation Status: {'Passed' if validation_status else 'Failed'}")

            return validation_status

        except Exception as e:
            logger.exception(f"Error during validation: {e}")
            raise e
