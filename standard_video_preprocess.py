# ----------------------------------------------------------------------------------
# Corrected PT Exercise Videos Preprocess, Upload to S3, Insert to Snowflake
# ----------------------------------------------------------------------------------

import os
import json
import time
import boto3
import cv2
import mediapipe as mp
import snowflake.connector
from dotenv import load_dotenv

# -----------------------------------------
# AWS S3 and Snowflake Configuration
# -----------------------------------------
load_dotenv()

s3_client = boto3.client("s3")

PT_VIDEO_BUCKET = os.getenv("PT_EXERCISE_VIDEO_BUCKET")
KEYPOINTS_BUCKET = os.getenv("PT_EXERCISE_KEYPOINTS_BUCKET")

# Snowflake Configuration
SNOWFLAKE_CONFIG = {
    "user": os.getenv("SNOWFLAKE_USERNAME"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
    "database": os.getenv("SNOWFLAKE_DATABASE"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA"),
}

# Local temporary folder for videos and keypoints
local_video_dir = "./temp_videos"
os.makedirs(local_video_dir, exist_ok=True)

# -----------------------------------------
# Functions for Video Processing and S3 Upload
# -----------------------------------------

def download_video_from_s3(video_file, local_video_path):
    """ Download video from S3 to local directory. """
    s3_client.download_file(PT_VIDEO_BUCKET, video_file, local_video_path)
    print(f"Downloaded {video_file} from S3.")

def extract_pose_keypoints(video_file, local_video_path, output_video_path):
    """處理影片以提取姿勢關鍵點，並儲存處理後的影片。"""
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    cap = cv2.VideoCapture(local_video_path)
    keypoints_data = []

    # 取得影片的基本資訊
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 初始化影片寫入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 編碼
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            frame_keypoints = {
                landmark.name: {
                    "x": results.pose_landmarks.landmark[landmark].x,
                    "y": results.pose_landmarks.landmark[landmark].y,
                    "z": results.pose_landmarks.landmark[landmark].z, 
                    'visibility': results.pose_landmarks.landmark[landmark].visibility
                } for landmark in mp_pose.PoseLandmark
            }
            keypoints_data.append(frame_keypoints)

            # 在影像上繪製姿勢標記
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

        # 寫入處理後的影格
        out.write(frame)

        # # 顯示處理中的影像
        # cv2.imshow('MediaPipe Pose', cv2.flip(frame, 1))
        # if cv2.waitKey(1) & 0xFF == 27:  # 按下 'Esc' 鍵退出
        #     break

    # 釋放資源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return keypoints_data


def upload_keypoints_to_s3(local_json_path, video_file):
    """ Upload the generated keypoints JSON file to S3. """
    s3_json_path = video_file.replace(".mp4", "_keypoints.json")
    s3_client.upload_file(local_json_path, KEYPOINTS_BUCKET, s3_json_path)
    print(f"Uploaded {s3_json_path} to {KEYPOINTS_BUCKET}.")


def process_videos(output_video_path):
    """ Download videos from S3, process them to extract pose keypoints, and upload to S3. """
    try:
        response = s3_client.list_objects_v2(Bucket=PT_VIDEO_BUCKET)
        print("S3 Connection Successful!\n")
        if "Contents" not in response:
            print("No video files found in the S3 bucket.")
            return

        video_files = [obj["Key"] for obj in response["Contents"] if obj["Key"].endswith(".mp4")]
        print(f"Found {len(video_files)} videos in {PT_VIDEO_BUCKET}.\n")

    except Exception as e:
        print("Error:", e)
        return

    start_time = time.time()

    for video_file in video_files:
        local_video_path = os.path.join(local_video_dir, os.path.basename(video_file))

        download_video_from_s3(video_file, local_video_path)

        keypoints_data = extract_pose_keypoints(video_file, local_video_path, output_video_path)

        # Save keypoints as JSON locally
        json_filename = os.path.splitext(os.path.basename(video_file))[0] + "_keypoints.json"
        local_json_path = os.path.join(local_video_dir, json_filename)

        with open(local_json_path, "w") as f:
            json.dump(keypoints_data, f, indent=4)

        upload_keypoints_to_s3(local_json_path, video_file)

        # Clean up local files
        os.remove(local_video_path)
        os.remove(local_json_path)

    end_time = time.time()
    print(f"Total video processing time: {end_time - start_time:.2f} s")


# -----------------------------------------
# Functions for Snowflake Insertion
# -----------------------------------------

def connect_snowflake():
    """ Connect to Snowflake database. """
    return snowflake.connector.connect(**SNOWFLAKE_CONFIG)


def create_snowflake_table():
    """ Create table in Snowflake if it does not exist. """
    conn = connect_snowflake()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS pose_keypoints_new (
            video_name STRING,
            frame_number INT,
            keypoint_name STRING,
            x FLOAT,
            y FLOAT,
            z FLOAT, 
            visibility FLOAT
        )
    """)
    conn.commit()
    cur.close()
    conn.close()


def insert_keypoints_to_snowflake(json_data, video_name):
    """ Insert the keypoints data into Snowflake. """
    conn = connect_snowflake()
    cur = conn.cursor()

    for frame_number, frame_keypoints in enumerate(json_data):
        for keypoint_name, coords in frame_keypoints.items():
            cur.execute("""
                INSERT INTO pose_keypoints_new (video_name, frame_number, keypoint_name, x, y, z, visibility)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (video_name, frame_number, keypoint_name, coords["x"], coords["y"], coords["z"], coords["visibility"]))

    conn.commit()
    cur.close()
    conn.close()


def process_keypoints_to_snowflake():
    """ Download keypoints JSON from S3 and insert into Snowflake. """
    try:
        response = s3_client.list_objects_v2(Bucket=KEYPOINTS_BUCKET)
        print("S3 Connection Successful!\n")
        if "Contents" not in response:
            print("No keypoints files found in the S3 bucket.")
            return

        json_files = [obj["Key"] for obj in response.get("Contents", []) if obj["Key"].endswith(".json")]
        print(f"Found {len(json_files)} keypoints files.\n")

    except Exception as e:
        print("Error:", e)
        return

    start_time = time.time()

    for json_file in json_files:
        obj = s3_client.get_object(Bucket=KEYPOINTS_BUCKET, Key=json_file)
        keypoints_data = json.load(obj["Body"])

        video_name = json_file.replace("_keypoints.json", "")
        insert_keypoints_to_snowflake(keypoints_data, video_name)

    end_time = time.time()
    print(f"Total Snowflake data insertion time: {end_time - start_time:.2f} s")

if __name__ == "__main__":
    # main()
    output_video_path = 'lower_extremity/output_correct_video.mp4'

    """ Main function to process videos and insert data into Snowflake. """
    create_snowflake_table()  # Ensure the table exists

    # Process Videos
    process_videos(output_video_path)

    # Process Keypoints and insert into Snowflake
    process_keypoints_to_snowflake()
