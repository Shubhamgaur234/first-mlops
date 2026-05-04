import os
from datetime import date

# =========================
# DATABASE CONFIG
# =========================
DATABASE_NAME = "Proj"
MONGODB_URL_KEY = "MONGODB_URL"
COLLECTION_NAME = "loan_data"

# =========================
# PIPELINE CONFIG
# =========================
PIPELINE_NAME: str = "loan_prediction_pipeline"
ARTIFACT_DIR: str = "artifact"

# =========================
# FILE NAMES
# =========================
MODEL_FILE_NAME = "model.pkl"
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"

FILE_NAME: str = "data.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")

# =========================
# TARGET
# =========================
TARGET_COLUMN = "Status"
CURRENT_YEAR = date.today().year

# =========================
# AWS CONFIG
# =========================
AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME = "us-east-1"

# =========================
# DATA INGESTION
# =========================
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.25

# 🔥 IMPORTANT FIX (your error was here)
DATA_INGESTION_COLLECTION_NAME: str = COLLECTION_NAME

# =========================
# DATA VALIDATION
# =========================
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME: str = "report.yaml"

# =========================
# DATA TRANSFORMATION
# =========================
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

# =========================
# MODEL TRAINER (XGBOOST)
# =========================
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"

MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")

# XGBoost parameters
MODEL_TRAINER_N_ESTIMATORS = 200
MODEL_TRAINER_MAX_DEPTH = 6
MODEL_TRAINER_LEARNING_RATE = 0.1
MODEL_TRAINER_SUBSAMPLE = 0.8
MODEL_TRAINER_RANDOM_STATE = 42

# =========================
# MODEL EVALUATION
# =========================
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02

# =========================
# MODEL PUSHER
# =========================
MODEL_BUCKET_NAME = "my-shubham-mlopsproj"
MODEL_PUSHER_S3_KEY = "model-registry"

# =========================
# APP CONFIG
# =========================
APP_HOST = "0.0.0.0"
APP_PORT = 5000