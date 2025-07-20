import os

from dotenv import load_dotenv

from src.paths import PARENT_DIR


load_dotenv(PARENT_DIR/'.env')


HOPSWORKS_PROJECT_NAME = 'taxidemanpredictor'

try:
    HOPSWORKS_API_KEY = os.environ['HOPSWORKS_API_KEY']
except:
    raise Exception('Create an .env file on the project root with HOPSWORKS_API_KEY')

FEATURE_GROUP_NAME = 'time_series_hourly_feature_group'
FEATURE_GROUP_VERSION = 1