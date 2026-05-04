import json
import sys
import os
import pandas as pd

from pandas import DataFrame

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml_file
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.constants import SCHEMA_FILE_PATH


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    # 🔥 FIXED: extract column names properly
    def get_schema_columns(self):
        schema_cols = []

        for col in self._schema_config["columns"]:
            if isinstance(col, dict):
                schema_cols.append(list(col.keys())[0])
            else:
                schema_cols.append(col)

        return schema_cols

    # 🔥 VALIDATE COLUMN EXISTENCE
    def validate_columns(self, df: DataFrame) -> bool:
        try:
            # remove MongoDB _id
            df = df.drop(columns=["_id"], errors="ignore")

            schema_cols = self.get_schema_columns()

            print("\n🔍 DEBUG INFO")
            print("Schema columns:", schema_cols)
            print("DF columns:", df.columns.tolist())

            missing_cols = [col for col in schema_cols if col not in df.columns]

            if missing_cols:
                print("❌ Missing columns:", missing_cols)
                logging.info(f"Missing columns: {missing_cols}")
                return False

            return True

        except Exception as e:
            raise MyException(e, sys)

    # 🔥 VALIDATE NUM + CAT FEATURES
    def validate_feature_columns(self, df: DataFrame) -> bool:
        try:
            df_cols = df.columns

            num_missing = [col for col in self._schema_config["numerical_columns"] if col not in df_cols]
            cat_missing = [col for col in self._schema_config["categorical_columns"] if col not in df_cols]

            if num_missing:
                print("❌ Missing numerical:", num_missing)
                logging.info(f"Missing numerical columns: {num_missing}")

            if cat_missing:
                print("❌ Missing categorical:", cat_missing)
                logging.info(f"Missing categorical columns: {cat_missing}")

            return False if num_missing or cat_missing else True

        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    # 🔥 MAIN VALIDATION PIPELINE
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            validation_error_msg = ""

            logging.info("Starting data validation")

            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            # 🔥 Drop _id safely
            train_df.drop(columns=["_id"], inplace=True, errors="ignore")
            test_df.drop(columns=["_id"], inplace=True, errors="ignore")

            # 🔥 1. Column validation
            if not self.validate_columns(train_df):
                validation_error_msg += "Columns are missing in training dataframe. "

            if not self.validate_columns(test_df):
                validation_error_msg += "Columns are missing in test dataframe. "

            # 🔥 2. Feature validation
            if not self.validate_feature_columns(train_df):
                validation_error_msg += "Feature columns missing in training dataframe. "

            if not self.validate_feature_columns(test_df):
                validation_error_msg += "Feature columns missing in test dataframe. "

            validation_status = len(validation_error_msg) == 0

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
                validation_report_file_path=self.data_validation_config.validation_report_file_path
            )

            # save report
            os.makedirs(os.path.dirname(self.data_validation_config.validation_report_file_path), exist_ok=True)

            with open(self.data_validation_config.validation_report_file_path, "w") as report_file:
                json.dump({
                    "validation_status": validation_status,
                    "message": validation_error_msg.strip()
                }, report_file, indent=4)

            logging.info("Data validation completed")
            return data_validation_artifact

        except Exception as e:
            raise MyException(e, sys)