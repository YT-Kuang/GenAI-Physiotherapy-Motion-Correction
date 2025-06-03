import numpy as np
import json
import os
import boto3
import ast
from dotenv import load_dotenv

# Import from own modules
from utils import (
    connect_s3, 
    connect_snowflake
)

# Load env variables
load_dotenv()

PATIENT_KEYPOINTS_BUCKET = os.getenv("PATIENT_KEYPOINTS_BUCKET")
NORMALIZED_BUCKET = os.getenv("RESULT_BUCKET")
OVERLAY_BUCKET = os.getenv("RESULT_BUCKET")

def fetch_keypoints_from_snowflake(video_name, table_name):
    """
    Fetch all keypoint data for a specific video from Snowflake. Returns:
      {
         frame_idx: {
             keypoint_name: np.array([x,y,z], dtype=float32),
             ...
         },
         ...
      }
    """
    conn = connect_snowflake()
    cursor = conn.cursor()
    
    query = f"""
    SELECT frame_number, keypoint_name, x, y, z
    FROM {table_name}
    WHERE video_name = '{video_name}';
    """
    cursor.execute(query)
    data = cursor.fetchall()
    
    keypoints_data = {}
    for row in data:
        frame_number, keypoint_name, x, y, z = row
        if frame_number not in keypoints_data:
            keypoints_data[frame_number] = {}
        keypoints_data[frame_number][keypoint_name] = np.array([x, y, z], dtype=np.float32)
    
    cursor.close()
    conn.close()
    
    return keypoints_data

def load_patient_keypoints_from_s3():
    """
    Load and process patient_keypoints.json from S3.
    Converts string values to np.array(dtype=np.float32).
    """
    s3_client = connect_s3()

    try:
        response = s3_client.list_objects_v2(Bucket=PATIENT_KEYPOINTS_BUCKET)
        print("[main] S3 Connection Successful!")
        if "Contents" not in response:
            print("[main] No keypoints files found in the S3 bucket.")
            return

        json_files = [obj["Key"] for obj in response.get("Contents", []) if obj["Key"].endswith(".json")]
        print(f"[main] Found {len(json_files)} keypoints files.")

    except Exception as e:
        print("[main] Error:", e)
        return

    for json_file in json_files:
        obj = s3_client.get_object(Bucket=PATIENT_KEYPOINTS_BUCKET, Key=json_file)
        patient_data_raw = json.load(obj["Body"])

    # Convert list to dictionary with integer frame indexes
    patient_keypoint_data = {
        int(frame_idx): {
            kp_name: np.array([float(kp_values["x"]), float(kp_values["y"]), float(kp_values["z"])], dtype=np.float32)
            for kp_name, kp_values in keypoints.items()
        }
        for frame_idx, keypoints in enumerate(patient_data_raw)  # Enumerate to convert list index to dict key
    }

    return patient_keypoint_data

# def load_normalized_data(skeleton_data_key, keypoints_order_key):
#     s3_client = boto3.client("s3")

#     # --- Load skeleton arrays JSON ---
#     keypoint_obj = s3_client.get_object(Bucket=NORMALIZED_BUCKET, Key=skeleton_data_key)
#     data = json.load(keypoint_obj["Body"])
#     correct_arr = np.array(data["correct"], dtype=np.float32)
#     patient_aligned_arr = np.array(data["patient_aligned"], dtype=np.float32)

#     # --- Load keypoint order TXT ---
#     kp_o = s3_client.get_object(Bucket=NORMALIZED_BUCKET, Key=keypoints_order_key)
#     txt = kp_o["Body"].read().decode("utf-8")
    
#     # Convert the Python-list-style string back to an actual list
#     kp_order = ast.literal_eval(txt)

#     return correct_arr, patient_aligned_arr, kp_order

def load_overlay_skeleton_animation_url():
    s3_client = connect_s3()

    response = s3_client.get_bucket_location(Bucket=OVERLAY_BUCKET)
    region = response['LocationConstraint'] or 'us-east-1'

    key = "lower_extremity_overlay/normalized_overlay_skeleton_animation_newVer.gif"
    url = f"https://{OVERLAY_BUCKET}.s3.{region}.amazonaws.com/{key}"
    
    return key, url