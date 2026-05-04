import os
import sys
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

from src.constants import TARGET_COLUMN
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data


class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifact,
        data_validation_artifact,
        data_transformation_config
    ):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_artifact = data_validation_artifact
        self.config = data_transformation_config

    # =========================
    # CLEAN DATA
    # =========================
    def clean_dataframe(self, df: pd.DataFrame):
        try:
            logging.info("Cleaning dataframe...")

            drop_cols = ["_id"]
            duplicate_target_col = TARGET_COLUMN.lower()
            if duplicate_target_col != TARGET_COLUMN and duplicate_target_col in df.columns:
                drop_cols.append(duplicate_target_col)

            df = df.drop(columns=drop_cols, errors="ignore")

            # Target cleaning
            df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
            df = df.dropna(subset=[TARGET_COLUMN])

            # Identify columns
            numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
            categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

            # Remove target
            if TARGET_COLUMN in numerical_cols:
                numerical_cols.remove(TARGET_COLUMN)
            if TARGET_COLUMN in categorical_cols:
                categorical_cols.remove(TARGET_COLUMN)

            # Fix types
            for col in numerical_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            for col in categorical_cols:
                df[col] = df[col].astype(str)

            return df, numerical_cols, categorical_cols

        except Exception as e:
            raise MyException(e, sys)

    # =========================
    # PREPROCESSOR
    # =========================
    def get_preprocessor(self, numerical_cols, categorical_cols):
        try:
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])

            preprocessor = ColumnTransformer([
                ("num", num_pipeline, numerical_cols),
                ("cat", cat_pipeline, categorical_cols)
            ])

            return preprocessor

        except Exception as e:
            raise MyException(e, sys)

    # =========================
    # MAIN FUNCTION
    # =========================
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting Data Transformation")

            # Load data from ingestion artifact
            train_df = pd.read_csv(self.data_ingestion_artifact.trained_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            # Clean
            train_df, num_cols, cat_cols = self.clean_dataframe(train_df)
            test_df, _, _ = self.clean_dataframe(test_df)

            # Split
            X_train = train_df.drop(columns=[TARGET_COLUMN])
            y_train = train_df[TARGET_COLUMN].astype(int)

            X_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN].astype(int)

            # Preprocessing
            preprocessor = self.get_preprocessor(num_cols, cat_cols)

            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            # =========================
            # SMOTE (only if valid)
            # =========================
            if len(np.unique(y_train)) > 1:
                logging.info("Applying SMOTE")
                smote = SMOTE(random_state=42)
                X_train_arr, y_train = smote.fit_resample(X_train_arr, y_train)
            else:
                logging.warning("SMOTE skipped (only one class)")

            # Combine
            train_arr = np.c_[X_train_arr, y_train]
            test_arr = np.c_[X_test_arr, y_test]

            # Save
            save_numpy_array_data(self.config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.config.transformed_test_file_path, test_arr)

            save_object(self.config.transformed_object_file_path, preprocessor)

            logging.info("Data Transformation completed successfully")

            return DataTransformationArtifact(
                transformed_object_file_path=self.config.transformed_object_file_path,
                transformed_train_file_path=self.config.transformed_train_file_path,
                transformed_test_file_path=self.config.transformed_test_file_path
            )

        except Exception as e:
            raise MyException(e, sys)
