import sys

import pandas as pd
from sklearn.metrics import f1_score

from src.constants import TARGET_COLUMN
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.s3_estimator import Proj1Estimator
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_object


class ModelEvaluation:
    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys) from e

    @staticmethod
    def normalize_target(values) -> pd.Series:
        target_values = pd.Series(values).reset_index(drop=True)
        mapped_values = target_values.map(
            {
                "no": 0,
                "yes": 1,
                "n": 0,
                "y": 1,
                "0": 0,
                "1": 1,
            }
        )
        numeric_values = pd.to_numeric(target_values, errors="coerce")
        normalized_values = numeric_values.fillna(mapped_values)
        return normalized_values.dropna().astype(int)

    def get_model_score(self, model, dataframe: pd.DataFrame) -> float:
        try:
            dataframe = dataframe.copy()
            dataframe[TARGET_COLUMN] = pd.to_numeric(
                dataframe[TARGET_COLUMN], errors="coerce"
            )
            dataframe = dataframe.dropna(subset=[TARGET_COLUMN])

            x = dataframe.drop(TARGET_COLUMN, axis=1)
            y_true = self.normalize_target(dataframe[TARGET_COLUMN])
            y_pred = self.normalize_target(model.predict(x))

            if len(y_true) != len(y_pred):
                raise Exception(
                    f"Prediction count {len(y_pred)} does not match target count {len(y_true)}"
                )

            return f1_score(y_true, y_pred, average="weighted", zero_division=0)
        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info("Starting model evaluation")

            test_df = pd.read_csv(
                self.data_ingestion_artifact.test_file_path,
                low_memory=False,
            )
            trained_model = load_object(
                self.model_trainer_artifact.trained_model_file_path
            )

            trained_model_score = self.get_model_score(trained_model, test_df)
            s3_model_path = self.model_eval_config.s3_model_key_path
            try:
                model_resolver = Proj1Estimator(
                    bucket_name=self.model_eval_config.bucket_name,
                    model_path=s3_model_path,
                )
            except Exception as e:
                logging.info(f"S3 model lookup skipped: {e}")
                model_resolver = None

            if model_resolver is None or not model_resolver.is_model_present(s3_model_path):
                logging.info("No model found in S3. Accepting trained model.")
                return ModelEvaluationArtifact(
                    is_model_accepted=True,
                    changed_accuracy=trained_model_score,
                    s3_model_path=s3_model_path,
                    trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                )

            s3_model = model_resolver.load_model()
            s3_model_score = self.get_model_score(s3_model, test_df)
            changed_accuracy = trained_model_score - s3_model_score
            is_model_accepted = (
                changed_accuracy >= self.model_eval_config.changed_threshold_score
            )

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                changed_accuracy=changed_accuracy,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
            )
            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys) from e
