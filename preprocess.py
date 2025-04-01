import os
import json
import time
import cv2
import mediapipe as mp
import tempfile
from dotenv import load_dotenv
from utils import connect_s3

load_dotenv()

s3_client = connect_s3()

# Configure your S3 bucket names here
PATIENT_VIDEO_BUCKET = os.getenv("PATIENT_VIDEO_BUCKET")
PATIENT_KEYPOINTS_BUCKET = os.getenv("PATIENT_KEYPOINTS_BUCKET")

# # Local temporary folder for videos and keypoints
# local_video_dir = "./temp_videos"
# os.makedirs(local_video_dir, exist_ok=True)

def download_video_from_s3(video_file):
    """
    Download a video file from S3 into a temporary file.
    Returns the temporary file object.
    """
    # s3_client.download_file(PATIENT_VIDEO_BUCKET, video_file, local_video_path)
    # print(f"[preprocess] Downloaded {video_file} from S3 to {local_video_path}.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        s3_client.download_file(PATIENT_VIDEO_BUCKET, video_file, temp_file.name)
        print(f"[preprocess] Downloaded {video_file} from S3 to a temporary file {temp_file.name}.")
        return temp_file.name

def extract_pose_keypoints(video_file, output_video_path):
    """
    Extracts pose keypoints from a video file object and and saves an optional annotated video.
    Returns keypoints data.
    """
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_file)
    keypoints_data = []

    # Get basic video info
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create a temporary file for processed video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video.name, fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                # Build a frame-level dict: { landmark_name: { x:..., y:..., z:..., visibility:... }, ... }
                frame_dict = {}
                for lm_id, lm_name in enumerate(mp_pose.PoseLandmark):
                    landmark = results.pose_landmarks.landmark[lm_name]
                    frame_dict[lm_name.name] = {
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z,
                        "visibility": landmark.visibility
                    }
                keypoints_data.append(frame_dict)

                # Optionally draw landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

            out.write(frame)

        # Release resources
        cap.release()
        out.release()
        # cv2.destroyAllWindows()

        # Move temp video to the final output path
        os.rename(temp_video.name, output_video_path)

    return keypoints_data, output_video_path

def upload_keypoints_to_s3(keypoints_data, video_file):
    """
    Upload the generated keypoints JSON file to S3.
    """
    json_filename_in_s3 = video_file.replace(".mp4", "_keypoints.json")
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as temp_json:
        json.dump(keypoints_data, temp_json, indent=4)
        temp_json.flush()  # Ensure all data is written before upload
        
        s3_client.upload_file(temp_json.name, PATIENT_KEYPOINTS_BUCKET, json_filename_in_s3)
        print(f"[preprocess] Uploaded {json_filename_in_s3} to s3://{PATIENT_KEYPOINTS_BUCKET}/")

        os.remove(temp_json.name)  # Cleanup

def upload_output_video_to_s3(output_video_path):
    """
    Upload the processed output video to S3.
    """
    # video_filename_in_s3 = os.path.basename(output_video_path)

    s3_client.upload_file(output_video_path, PATIENT_KEYPOINTS_BUCKET, output_video_path)
    print(f"[preprocess] Uploaded {output_video_path} to s3://{PATIENT_KEYPOINTS_BUCKET}/")

    os.remove(output_video_path)  # Cleanup

def process_videos(output_video_path):
    """
    Process videos: 
    - Download from S3
    - Extract pose keypoints
    - Upload keypoints JSON to S3
    - Upload processed video to S3
    """
    try:
        response = s3_client.list_objects_v2(Bucket=PATIENT_VIDEO_BUCKET)
        if "Contents" not in response:
            print("[preprocess] No video files found in the S3 bucket.")
            return

        # Filter only .mp4
        video_files = [obj["Key"] for obj in response["Contents"] if obj["Key"].endswith(".mp4")]
        print(f"[preprocess] Found {len(video_files)} .mp4 files in {PATIENT_VIDEO_BUCKET}.")
    except Exception as e:
        print("[preprocess] Error listing S3 objects:", e)
        return

    start_time = time.time()

    # for video_file in video_files:
    #     local_video_path = os.path.join(local_video_dir, os.path.basename(video_file))
    #     download_video_from_s3(video_file, local_video_path)

    #     # Extract keypoints
    #     keypoints_data = extract_pose_keypoints(local_video_path, output_video_path)

    #     # Save keypoints locally as JSON
    #     json_filename = os.path.splitext(os.path.basename(video_file))[0] + "_keypoints.json"
    #     local_json_path = os.path.join(local_video_dir, json_filename)
    #     with open(local_json_path, "w") as f:
    #         json.dump(keypoints_data, f, indent=4)

    #     # Upload JSON to S3
    #     upload_keypoints_to_s3(local_json_path, video_file)

    #     # Clean up local files
    #     os.remove(local_video_path)
    #     os.remove(local_json_path)

    for video_file in video_files:
        temp_video_path = download_video_from_s3(video_file)
        keypoints_data, processed_video_path = extract_pose_keypoints(temp_video_path, output_video_path)
        upload_keypoints_to_s3(keypoints_data, video_file)
        upload_output_video_to_s3(processed_video_path)

        os.remove(temp_video_path)  # Cleanup

    end_time = time.time()
    print(f"[preprocess] Total video processing time: {end_time - start_time:.2f} seconds.")
