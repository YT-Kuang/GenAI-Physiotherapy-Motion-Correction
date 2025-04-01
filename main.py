import os
import json
import time
import openai
import numpy as np
from dotenv import load_dotenv

# Import from own modules
from utils import (
    connect_s3, 
    connect_snowflake
)
from preprocess import process_videos
from normalized import (
    align_skeleton_to_standard,
    single_global_rotation,
    apply_global_rotation_to_dict,
    dictionary_to_frame_array,
    dtw_time_warping
)
from results import (
    draw_overlay_skeleton_animation,
    compute_3d_coordinate_rmse_for_keypoints,
    compute_knee_angle_rmse, 
    save_rmse_to_dataframe, 
    upload_dataframe_to_snowflake
)
from llm import (
    fetch_rmse_metrics_from_snowflake, 
    generate_physio_report
)

# Load env variables
load_dotenv()

# Load AWS S3 credentials
PATIENT_KEYPOINTS_BUCKET = os.getenv("PATIENT_KEYPOINTS_BUCKET")
OVERLAY_GIF = os.getenv("OVERLAY_ANIMATION_BUCKET")

# Load OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

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

def main():
    # 1) Process videos from S3
    process_videos(output_video_path="lower_extremity/output_patient_video.mp4")

    # 2) Load keypoints 
    correct_video_keypoints = fetch_keypoints_from_snowflake(
        'lower_extremity/resized_correct_video', 
        'pose_keypoints_new'
    )
    patient_video_keypoints = load_patient_keypoints_from_s3()

    # 3) Align each skeleton to a standard local frame
    aligned_correct = align_skeleton_to_standard(correct_video_keypoints)
    aligned_patient = align_skeleton_to_standard(patient_video_keypoints)

    # 4) Apply a single global rotation so that the "correct" orientation is closer to the patient's
    R_global = single_global_rotation(aligned_correct, aligned_patient)
    aligned_correct = apply_global_rotation_to_dict(aligned_correct, R_global)

    # 5) Convert each dictionary into a NumPy array
    correct_arr, correct_frames, kp_order = dictionary_to_frame_array(aligned_correct)
    patient_arr, patient_frames, _ = dictionary_to_frame_array(aligned_patient, kp_order)

    # 6) Time warp (DTW) the patient array to match the correct array length
    patient_aligned_arr = dtw_time_warping(patient_arr, correct_arr)

    start_time = time.time()

    # 7) Visualize both skeletons in one 3D animation (optional)
    draw_overlay_skeleton_animation(
        arr_correct=correct_arr,
        arr_patient=patient_aligned_arr,
        keypoints_order=kp_order,
        bucket_name = OVERLAY_GIF,
        s3_save_path = "lower_extremity/normalized_overlay_skeleton_animation_newVer.gif"
    )

    # 8) Compute numeric metrics
        # 3D coordinate RMSE for certain keypoints
    keypoints_of_interest = [
        "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
        "LEFT_ANKLE", "RIGHT_ANKLE", 'LEFT_HEEL', 'RIGHT_HEEL', 
        "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
    ]
    coordinate_rmses = compute_3d_coordinate_rmse_for_keypoints(
        correct_arr, patient_aligned_arr, keypoints_of_interest, kp_order
    )
    
    # print("\n[main] 3D RMSE for selected keypoints:")
    # for kp_name, rmse_val in zip(keypoints_of_interest, coordinate_rmses):
    #     print(f"   {kp_name}: {rmse_val:.4f}")

    # Knee angle + hip abduction angle
    knee_rmse, hip_abd_rmse = compute_knee_angle_rmse(correct_arr, patient_aligned_arr, kp_order)
    # print(f"\n[main] Knee Angle RMSE: {knee_rmse:.4f}")
    # print(f"[main] Hip Abduction Angle RMSE: {hip_abd_rmse:.4f}")

        # Convert RMSE results to DataFrame
    rmse_df = save_rmse_to_dataframe(coordinate_rmses, keypoints_of_interest, knee_rmse, hip_abd_rmse)
    
        # Upload the DataFrame to Snowflake
    upload_dataframe_to_snowflake(rmse_df)

    # 9) LLM
    rmse_metrics = fetch_rmse_metrics_from_snowflake()
    patient_info = {"age": 35, "height": 165, "weight": 60}
    report = generate_physio_report(patient_info, rmse_metrics)

    print("\n[main] Generated Physiotherapy Report:\n")
    print(json.dumps(report, indent=2))

    end_time = time.time()
    print(f"[main] Total time: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
