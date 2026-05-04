# from src.logger import logging
# #from src.exception import MyException
# import sys
# logging.debug("This is a debug message.")




# try:
#     a = 1+'Z'
# except Exception as e:
#      logging.info(e)
#      raise MyException(e, sys) from e

from dotenv import load_dotenv
load_dotenv()

from src.pipeline.training_pipeline import TrainPipeline

pipeline = TrainPipeline()
pipeline.run_pipeline()