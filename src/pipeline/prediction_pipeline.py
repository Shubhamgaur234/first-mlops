import os
import sys
from dataclasses import asdict, dataclass
from glob import glob
from typing import Optional

import pandas as pd

from src.constants import (
    AWS_ACCESS_KEY_ID_ENV_KEY,
    AWS_SECRET_ACCESS_KEY_ENV_KEY,
    MODEL_FILE_NAME,
)
from src.entity.config_entity import ModelPusherConfig
from src.entity.s3_estimator import Proj1Estimator
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_object


@dataclass
class LoanData:
    ID: Optional[float] = None
    year: Optional[float] = None
    loan_limit: Optional[str] = None
    Gender: Optional[str] = None
    approv_in_adv: Optional[str] = None
    loan_type: Optional[str] = None
    loan_purpose: Optional[str] = None
    Credit_Worthiness: Optional[str] = None
    open_credit: Optional[str] = None
    business_or_commercial: Optional[str] = None
    loan_amount: Optional[float] = None
    rate_of_interest: Optional[float] = None
    Interest_rate_spread: Optional[float] = None
    Upfront_charges: Optional[float] = None
    term: Optional[float] = None
    Neg_ammortization: Optional[str] = None
    interest_only: Optional[str] = None
    lump_sum_payment: Optional[str] = None
    property_value: Optional[float] = None
    construction_type: Optional[str] = None
    occupancy_type: Optional[str] = None
    Secured_by: Optional[str] = None
    total_units: Optional[str] = None
    income: Optional[float] = None
    credit_type: Optional[str] = None
    Credit_Score: Optional[float] = None
    co_applicant_credit_type: Optional[str] = None
    age: Optional[str] = None
    submission_of_application: Optional[str] = None
    LTV: Optional[float] = None
    Region: Optional[str] = None
    Security_Type: Optional[str] = None
    dtir1: Optional[float] = None

    def get_loan_input_data_frame(self) -> pd.DataFrame:
        try:
            data = asdict(self)
            data["co-applicant_credit_type"] = data.pop("co_applicant_credit_type")
            return pd.DataFrame([data])
        except Exception as e:
            raise MyException(e, sys) from e


class LoanDataClassifier:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None

    def _latest_local_model_path(self) -> str:
        model_paths = glob(
            os.path.join("artifact", "*", "model_trainer", "trained_model", MODEL_FILE_NAME)
        )
        if not model_paths:
            raise Exception("No trained model found. Run /train first.")
        return max(model_paths, key=os.path.getmtime)

    def _load_model(self):
        if self.model is not None:
            return self.model

        if self.model_path:
            self.model = load_object(self.model_path)
            return self.model

        if os.getenv(AWS_ACCESS_KEY_ID_ENV_KEY) and os.getenv(AWS_SECRET_ACCESS_KEY_ENV_KEY):
            try:
                config = ModelPusherConfig()
                estimator = Proj1Estimator(
                    bucket_name=config.bucket_name,
                    model_path=config.s3_model_key_path,
                )
                if estimator.is_model_present(config.s3_model_key_path):
                    self.model = estimator.load_model()
                    return self.model
            except Exception as e:
                logging.info(f"S3 model load skipped: {e}")

        self.model = load_object(self._latest_local_model_path())
        return self.model

    def predict(self, dataframe: pd.DataFrame):
        try:
            model = self._load_model()
            return model.predict(dataframe=dataframe)
        except Exception as e:
            raise MyException(e, sys) from e
