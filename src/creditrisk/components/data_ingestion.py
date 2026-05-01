import os
import shutil
from src.creditrisk.utils import logger
from src.creditrisk.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_data(self):
        """
        Downloads the data file. Since this dataset requires Kaggle auth,
        we copy it from the local data/raw/ directory instead.
        """
        dest = str(self.config.local_data_file)
        if not os.path.exists(dest):
            src = os.path.join("data", "raw", "credit_risk.csv")
            if os.path.exists(src):
                shutil.copy(src, dest)
                logger.info(f"Data copied from {src} to {dest}")
            else:
                raise FileNotFoundError(
                    f"Source file not found: {src}\n"
                    f"Please place the credit risk CSV at: data/raw/credit_risk.csv"
                )
        else:
            logger.info(f"File already exists at: {dest}")

    def extract_zip_file(self):
        """
        No-op for CSV datasets — the file is already extracted.
        Validates that the file is accessible in the unzip directory.
        """
        unzip_path = str(self.config.unzip_dir)
        csv_path = os.path.join(unzip_path, "credit_risk.csv")
        if os.path.exists(csv_path):
            logger.info(f"Data file ready at: {csv_path}")
        else:
            raise FileNotFoundError(f"Expected data file not found at: {csv_path}")
