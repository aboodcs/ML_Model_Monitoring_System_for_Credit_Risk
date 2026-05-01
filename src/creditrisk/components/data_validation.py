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
        """
        try:
            validation_status = True

            # Read the extracted data file
            data_file = os.path.join(self.config.unzip_data_dir, "credit_risk.csv")
            # Skip the first row (extra header) and drop the ID column
            df = pd.read_csv(data_file, skiprows=1)
            df = df.drop(columns=df.columns[0], axis=1)

            # Get expected columns from schema
            all_schema_columns = self.config.all_schema.keys()

            # Get actual columns from the dataframe
            df_columns = df.columns.tolist()

            # Validate each column
            for col in df_columns:
                if col not in all_schema_columns:
                    validation_status = False
                    logger.info(f"Column '{col}' not found in schema")

            if validation_status:
                logger.info("All columns validated successfully against schema")

            # Write validation status to file
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation Status: {'Passed' if validation_status else 'Failed'}")

            return validation_status

        except Exception as e:
            logger.exception(f"Error during validation: {e}")
            raise e
