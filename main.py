import os
import time
import openai
import numpy as np
from dotenv import load_dotenv

# Import from own modules
from preprocess import process_videos
from normalized import normalized_process
from load_data import (
    fetch_keypoints_from_snowflake,
    load_patient_keypoints_from_s3,
    load_overlay_skeleton_animation_url
)
from results import (
    draw_overlay_skeleton_animation,
    compute_3d_coordinate_rmse_for_keypoints,
    compute_knee_angle_rmse,
    upload_rsme_to_snowflake
)
from llm import (
    fetch_rmse_metrics_from_snowflake, 
    generate_physio_report
)

# Load env variables
load_dotenv()

# Load AWS S3 credentials
PATIENT_KEYPOINTS_BUCKET = os.getenv("PATIENT_KEYPOINTS_BUCKET")
OVERLAY_BUCKET = os.getenv("RESULT_BUCKET")
GEN_REPORT_BUCKET = os.getenv("RESULT_BUCKET")
NORMALIZED_BUCKET = os.getenv("RESULT_BUCKET")

# Load OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def main(age, weight, height):

    total_start_time = time.time()

    # 1) Process videos from S3
    process_videos(output_video_path="lower_extremity/output_patient_video.mp4")

    # 2) Load keypoints 
    correct_video_keypoints = fetch_keypoints_from_snowflake(
        'lower_extremity/resized_correct_video', 
        'pose_keypoints_new'
    )
    patient_video_keypoints = load_patient_keypoints_from_s3()

    skeleton_data_s3_key = "lower_extremity_keypoint/corr_patientaligned_keypoint.json"
    keypoints_order_s3_key = "keypoint_order.txt"
    correct_arr, patient_aligned_arr, kp_order = normalized_process(correct_video_keypoints, 
                                                                    patient_video_keypoints, 
                                                                    NORMALIZED_BUCKET,
                                                                    skeleton_data_s3_key,
                                                                    keypoints_order_s3_key)
    
    result_start_time = time.time()
    # 7) Visualize both skeletons in one 3D animation (optional)
    s3_object_key, overlay_skeleton_animation_url = load_overlay_skeleton_animation_url()
    draw_overlay_skeleton_animation(
        arr_correct=correct_arr,
        arr_patient=patient_aligned_arr,
        keypoints_order=kp_order,
        bucket_name = OVERLAY_BUCKET,
        s3_save_path = s3_object_key
    )

    # 8) Compute numeric metrics
        # 3D coordinate RMSE for certain keypoints
    # keypoints_of_interest = [
    #     "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
    #     "LEFT_ANKLE", "RIGHT_ANKLE", 'LEFT_HEEL', 'RIGHT_HEEL', 
    #     "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
    # ]
    coordinate_rmses = compute_3d_coordinate_rmse_for_keypoints(
        correct_arr, patient_aligned_arr, kp_order, kp_order
    )
    
    # print("\n[main] 3D RMSE for selected keypoints:")
    # for kp_name, rmse_val in zip(keypoints_of_interest, coordinate_rmses):
    #     print(f"   {kp_name}: {rmse_val:.4f}")

    # Knee angle + hip abduction angle
    knee_rmse, hip_abd_rmse = compute_knee_angle_rmse(correct_arr, patient_aligned_arr, kp_order)
    # print(f"\n[main] Knee Angle RMSE: {knee_rmse:.4f}")
    # print(f"[main] Hip Abduction Angle RMSE: {hip_abd_rmse:.4f}")

        # Convert RMSE results to DataFrame
    # rmse_df = save_rmse_to_dataframe(coordinate_rmses, keypoints_of_interest, knee_rmse, hip_abd_rmse)
    
        # Upload the DataFrame to Snowflake
    upload_rsme_to_snowflake(coordinate_rmses,
                             kp_order,
                             knee_rmse, hip_abd_rmse
                            )
    result_end_time = time.time()

    llm_start_time = time.time()
    # 9) LLM
    rmse_metrics = fetch_rmse_metrics_from_snowflake()
    patient_info = {"age": age, "height": height, "weight": weight}
    report_output = generate_physio_report(patient_info,
                                           rmse_metrics,
                                           overlay_skeleton_animation_url,
                                           GEN_REPORT_BUCKET,
                                           "lower_extremity_gen_report/physio_feedback.json")
    
    # Unpack
    # physio_feedback_dict = report_output["report_dict"]
    llm_prompt = report_output["prompt"]
    physio_feedback_json_str = report_output["report_json"]
    # report_s3_url = report_output["report_s3_path"]

    # gen_report = json.dumps(report, indent=2)
    llm_end_time = time.time()

    print("\n[main] Generated Physiotherapy Report:\n")
    print(physio_feedback_json_str)

    # file_name = 'gen_report.json'

    # # Write the JSON data to a file
    # with open(file_name, 'w') as json_file:
    #     json_file.write(gen_report)
    # print(f"[main] Report saved to {file_name}")

    total_end_time = time.time()

    print(f"[main] Skeleton drawing / RMSE caculation time: {result_end_time - result_start_time:.2f} seconds.")
    print(f"[main] Feedback generation time: {llm_end_time - llm_start_time:.2f} seconds.")
    print(f"[main] Total time: {total_end_time - total_start_time:.2f} seconds.")

    return llm_prompt, physio_feedback_json_str

# if __name__ == "__main__":
#     main()
