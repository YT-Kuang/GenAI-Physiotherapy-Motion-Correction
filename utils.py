import snowflake.connector
import os
import boto3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SNOWFLAKE_CONFIG = {
    "user": os.getenv("SNOWFLAKE_USERNAME"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
    "database": os.getenv("SNOWFLAKE_DATABASE"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA"),
}

def connect_snowflake():
    return snowflake.connector.connect(**SNOWFLAKE_CONFIG)

def connect_s3():
    return boto3.client("s3")