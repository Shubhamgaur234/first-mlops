import sys
import os

from src.exception import MyException
from src.logger import logging
from src.constants import AWS_ACCESS_KEY_ID_ENV_KEY, AWS_SECRET_ACCESS_KEY_ENV_KEY
from src.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from src.entity.config_entity import ModelPusherConfig
from src.entity.s3_estimator import Proj1Estimator


class ModelPusher:
    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        """
        :param model_evaluation_artifact: Output reference of data evaluation artifact stage
        :param model_pusher_config: Configuration for model pusher
        """
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model pusher
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_model_pusher method of ModelTrainer class")

        try:
            print("------------------------------------------------------------------------------------------------")
            logging.info("Uploading artifacts folder to s3 bucket")

            if not self.model_evaluation_artifact.is_model_accepted:
                raise Exception("Cannot push model because it was not accepted.")

            if not os.getenv(AWS_ACCESS_KEY_ID_ENV_KEY) or not os.getenv(AWS_SECRET_ACCESS_KEY_ENV_KEY):
                logging.info("AWS credentials are not set. Skipping S3 model push.")
                return ModelPusherArtifact(bucket_name=self.model_pusher_config.bucket_name,
                                           s3_model_path=self.model_pusher_config.s3_model_key_path)

            proj1_estimator = Proj1Estimator(bucket_name=self.model_pusher_config.bucket_name,
                                             model_path=self.model_pusher_config.s3_model_key_path)
            
            logging.info("Uploading new model to S3 bucket....")
            proj1_estimator.save_model(from_file=self.model_evaluation_artifact.trained_model_path)
            model_pusher_artifact = ModelPusherArtifact(bucket_name=self.model_pusher_config.bucket_name,
                                                        s3_model_path=self.model_pusher_config.s3_model_key_path)

            logging.info("Uploaded artifacts folder to s3 bucket")
            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            logging.info("Exited initiate_model_pusher method of ModelTrainer class")
            
            return model_pusher_artifact
        except Exception as e:
            raise MyException(e, sys) from e
