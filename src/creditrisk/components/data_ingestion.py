import os
import urllib.request as request
from src.creditrisk.utils import logger
from src.creditrisk.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_data(self):
        """
        Downloads the train CSV directly from GitHub using urllib (no auth needed).
        """
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url=str(self.config.source_URL),
                filename=str(self.config.local_data_file)
            )
            logger.info(f"File downloaded successfully and saved at: {filename}")
        else:
            logger.info(f"File already exists at: {self.config.local_data_file}")

    def extract_zip_file(self):
        """
        No-op — the file is already a plain CSV, nothing to extract.
        Just validates the file exists.
        """
        if os.path.exists(str(self.config.local_data_file)):
            logger.info(f"Data file ready at: {self.config.local_data_file}")
        else:
            raise FileNotFoundError(
                f"Expected data file not found at: {self.config.local_data_file}"
            )
