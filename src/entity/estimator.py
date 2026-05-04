import sys
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline

from src.exception import MyException
from src.logger import logging


# =========================================
# Target Mapping Class
# =========================================
class TargetValueMapping:
    def __init__(self):
        # 🔥 Standard mapping
        self.no: int = 0
        self.yes: int = 1

    def _asdict(self):
        return self.__dict__

    def reverse_mapping(self):
        """
        Convert numeric prediction → original label
        Example: 0 → no, 1 → yes
        """
        mapping = self._asdict()
        return {v: k for k, v in mapping.items()}


# =========================================
# Model Wrapper Class
# =========================================
class MyModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        """
        :param preprocessing_object: Preprocessing pipeline
        :param trained_model_object: Trained ML model
        """
        try:
            self.preprocessing_object = preprocessing_object
            self.trained_model_object = trained_model_object
        except Exception as e:
            raise MyException(e, sys)

    def predict(self, dataframe: pd.DataFrame):
        """
        Accepts raw dataframe → applies preprocessing → returns prediction
        """
        try:
            logging.info("🔹 Starting prediction process")

            # ✅ Input validation
            if not isinstance(dataframe, pd.DataFrame):
                raise Exception("Input must be a pandas DataFrame")

            if dataframe.shape[0] == 0:
                raise Exception("Empty dataframe provided")

            logging.info(f"Input shape: {dataframe.shape}")

            # =====================================
            # Step 1: Apply preprocessing
            # =====================================
            transformed_feature = self.preprocessing_object.transform(dataframe)

            logging.info("✅ Data transformed successfully")

            # =====================================
            # Step 2: Model prediction
            # =====================================
            preds = self.trained_model_object.predict(transformed_feature)

            logging.info("✅ Prediction completed")

            # =====================================
            # Step 3: Convert to original labels
            # =====================================
            mapper = TargetValueMapping().reverse_mapping()
            preds = pd.Series(preds).map(mapper)

            return preds

        except Exception as e:
            logging.error("❌ Error occurred in predict method", exc_info=True)
            raise MyException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"