import sys

from sklearn.metrics import f1_score, precision_score, recall_score
from xgboost import XGBClassifier

from src.entity.artifact_entity import (
    ClassificationMetricArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from src.entity.config_entity import ModelTrainerConfig
from src.entity.estimator import MyModel
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, load_object, save_object


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise MyException(e, sys) from e

    def get_model_object(self) -> XGBClassifier:
        try:
            return XGBClassifier(
                n_estimators=self.model_trainer_config.n_estimators,
                max_depth=self.model_trainer_config.max_depth,
                learning_rate=self.model_trainer_config.learning_rate,
                subsample=self.model_trainer_config.subsample,
                random_state=self.model_trainer_config.random_state,
                eval_metric="logloss",
            )
        except Exception as e:
            raise MyException(e, sys) from e

    def get_classification_metrics(self, y_true, y_pred) -> ClassificationMetricArtifact:
        try:
            return ClassificationMetricArtifact(
                f1_score=f1_score(y_true, y_pred, average="weighted", zero_division=0),
                precision_score=precision_score(y_true, y_pred, average="weighted", zero_division=0),
                recall_score=recall_score(y_true, y_pred, average="weighted", zero_division=0),
            )
        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Starting model training")

            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1].astype(int)
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1].astype(int)

            model = self.get_model_object()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            metric_artifact = self.get_classification_metrics(y_test, y_pred)

            if metric_artifact.f1_score < self.model_trainer_config.expected_accuracy:
                raise Exception(
                    "Model f1 score "
                    f"{metric_artifact.f1_score} is below expected score "
                    f"{self.model_trainer_config.expected_accuracy}"
                )

            preprocessing_object = load_object(
                self.data_transformation_artifact.transformed_object_file_path
            )
            model_object = MyModel(
                preprocessing_object=preprocessing_object,
                trained_model_object=model,
            )

            save_object(self.model_trainer_config.trained_model_file_path, model_object)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )

            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise MyException(e, sys) from e
