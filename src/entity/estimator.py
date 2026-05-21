import sys

import pandas as pd
from sklearn.pipeline import Pipeline

from src.exception import MyException
from src.logger import logging


class TargetValueMapping:
    def __init__(self):
        self.no: int = 0
        self.yes: int = 1

    def _asdict(self):
        return self.__dict__

    def reverse_mapping(self):
        """
        Convert numeric prediction to original label.
        Example: 0 -> no, 1 -> yes.
        """
        mapping = self._asdict()
        return {v: k for k, v in mapping.items()}


class MyModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        try:
            self.preprocessing_object = preprocessing_object
            self.trained_model_object = trained_model_object
        except Exception as e:
            raise MyException(e, sys) from e

    def predict(self, dataframe: pd.DataFrame):
        try:
            logging.info("Starting prediction process")

            if not isinstance(dataframe, pd.DataFrame):
                raise Exception("Input must be a pandas DataFrame")

            if dataframe.shape[0] == 0:
                raise Exception("Empty dataframe provided")

            logging.info(f"Input shape: {dataframe.shape}")

            transformed_feature = self.preprocessing_object.transform(dataframe)
            logging.info("Data transformed successfully")

            preds = self.trained_model_object.predict(transformed_feature)
            logging.info("Prediction completed")

            mapper = TargetValueMapping().reverse_mapping()
            return pd.Series(preds).map(mapper)

        except Exception as e:
            logging.error("Error occurred in predict method", exc_info=True)
            raise MyException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"
