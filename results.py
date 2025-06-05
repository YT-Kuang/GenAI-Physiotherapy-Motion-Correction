import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import subprocess
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from dotenv import load_dotenv
import os
from utils import (
    connect_s3, 
    connect_snowflake
)

s3_client = connect_s3()

def draw_overlay_skeleton_animation(arr_correct, arr_patient, keypoints_order, bucket_name, s3_save_path):
    """
    Overlays two 3D skeleton sequences (arr_correct, arr_patient) in a single animation.
    Each array: shape=(F, K, 3).
    """
    # Define a set of limbs to connect. You can adjust this list if needed.
    limbs = [
        ('NOSE', 'LEFT_EYE_INNER'), ('LEFT_EYE_INNER', 'LEFT_EYE'), ('LEFT_EYE', 'LEFT_EYE_OUTER'),
        ('NOSE', 'RIGHT_EYE_INNER'), ('RIGHT_EYE_INNER', 'RIGHT_EYE'), ('RIGHT_EYE', 'RIGHT_EYE_OUTER'),
        ('NOSE', 'MOUTH_LEFT'), ('NOSE', 'MOUTH_RIGHT'),

        ('LEFT_EAR', 'LEFT_SHOULDER'), ('RIGHT_EAR', 'RIGHT_SHOULDER'),

        ('LEFT_SHOULDER', 'LEFT_ELBOW'), ('LEFT_ELBOW', 'LEFT_WRIST'),
        ('RIGHT_SHOULDER', 'RIGHT_ELBOW'), ('RIGHT_ELBOW', 'RIGHT_WRIST'),

        ('LEFT_SHOULDER', 'RIGHT_SHOULDER'), ('LEFT_HIP', 'RIGHT_HIP'),

        ('LEFT_HIP', 'LEFT_KNEE'), ('LEFT_KNEE', 'LEFT_ANKLE'),
        ('RIGHT_HIP', 'RIGHT_KNEE'), ('RIGHT_KNEE', 'RIGHT_ANKLE'),

        ('LEFT_ANKLE', 'LEFT_HEEL'), ('RIGHT_ANKLE', 'RIGHT_HEEL'),
        ('LEFT_HEEL', 'LEFT_FOOT_INDEX'), ('RIGHT_HEEL', 'RIGHT_FOOT_INDEX')
    ]
    kp_index_map = {name: i for i, name in enumerate(keypoints_order)}

    F = arr_correct.shape[0]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20, azim=-60)

    # Collect all coords for bounding box
    x_vals = np.concatenate([arr_correct[..., 0].ravel(), arr_patient[..., 0].ravel()])
    y_vals = np.concatenate([arr_correct[..., 1].ravel(), arr_patient[..., 1].ravel()])
    z_vals = np.concatenate([arr_correct[..., 2].ravel(), arr_patient[..., 2].ravel()])

    x_min, x_max = np.nanmin(x_vals), np.nanmax(x_vals)
    y_min, y_max = np.nanmin(y_vals), np.nanmax(y_vals)
    z_min, z_max = np.nanmin(z_vals), np.nanmax(z_vals)

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    scat_correct = ax.scatter([], [], [], marker='o', s=20, c='b')
    scat_patient = ax.scatter([], [], [], marker='o', s=20, c='orange')

    # Create line objects for skeleton connections
    lines_correct = [ax.plot([], [], [], 'b', lw=2)[0] for _ in limbs]
    lines_patient = [ax.plot([], [], [], 'orange', lw=2)[0] for _ in limbs]

    def update(frame_idx):
        ccoords = arr_correct[frame_idx]  # (K,3)
        pcoords = arr_patient[frame_idx]  # (K,3)

        scat_correct._offsets3d = (ccoords[:, 0], ccoords[:, 1], ccoords[:, 2])
        scat_patient._offsets3d = (pcoords[:, 0], pcoords[:, 1], pcoords[:, 2])

        for i, (start, end) in enumerate(limbs):
            if start in kp_index_map and end in kp_index_map:
                s_i, e_i = kp_index_map[start], kp_index_map[end]
                lines_correct[i].set_data([ccoords[s_i, 0], ccoords[e_i, 0]],
                                          [ccoords[s_i, 1], ccoords[e_i, 1]])
                lines_correct[i].set_3d_properties([ccoords[s_i, 2], ccoords[e_i, 2]])

                lines_patient[i].set_data([pcoords[s_i, 0], pcoords[e_i, 0]],
                                          [pcoords[s_i, 1], pcoords[e_i, 1]])
                lines_patient[i].set_3d_properties([pcoords[s_i, 2], pcoords[e_i, 2]])

        return (scat_correct, scat_patient) + tuple(lines_correct) + tuple(lines_patient)

    ani = animation.FuncAnimation(fig, update, frames=F, interval=30, blit=False)

    # Save the animation to a temporary in-memory file
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as temp_file:
        temp_path = temp_file.name
        ani.save(temp_path, writer=PillowWriter(fps=30))

    s3_client.upload_file(temp_path, bucket_name, s3_save_path)
    
    print(f"[result] Overlay skeleton animation uploaded to s3://{bucket_name}/{s3_save_path}")

    # # Save ani to local
    # ani.save(save_path, writer=PillowWriter(fps=30))
    # print(f"[result] Overlay skeleton animation saved at: {save_path}")


def compute_3d_coordinate_rmse_for_keypoints(correct_arr, patient_arr, keypoints, kp_order):
    """
    Compute the 3D RMSE for a subset of keypoints.
      - correct_arr, patient_arr: shape = (F, K, 3)
      - keypoints: list of keypoint names to measure
      - kp_order: the ordered list of keypoints (so we can index them)
    Returns: list of RMSE values in the same order as `keypoints`.
    """
    name_to_idx = {name: i for i, name in enumerate(kp_order)}
    rmses = []
    for kp_name in keypoints:
        if kp_name not in name_to_idx:
            rmses.append(np.nan)
            continue
        idx = name_to_idx[kp_name]
        diff = patient_arr[:, idx, :] - correct_arr[:, idx, :]
        diff_sq = diff ** 2
        mse = np.nanmean(diff_sq)
        rmse = np.sqrt(mse)
        rmses.append(rmse)
    return rmses


def angle_between_vectors(v1, v2):
    """
    Returns the angle between two 3D vectors (in degrees).
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    cos_val = np.dot(v1, v2) / (norm1 * norm2)
    cos_val = np.clip(cos_val, -1.0, 1.0)
    return np.degrees(np.arccos(cos_val))

def compute_hip_abduction_angle(hip, knee):
    """
    Hip abduction angle = angle between vector(hip->knee) and the vertical axis (0,1,0).
    """
    v1 = knee - hip
    v2 = np.array([0, 1, 0], dtype=np.float32)
    return angle_between_vectors(v1, v2)

def compute_knee_angle(hip, knee, ankle):
    """
    Knee flexion angle = angle between vectors (hip->knee) and (ankle->knee).
    """
    v1 = hip - knee
    v2 = ankle - knee
    return angle_between_vectors(v1, v2)

def compute_knee_angle_rmse(correct_arr, patient_arr, kp_order):
    """
    Computes the average RMSE of left & right knee angles, plus hip abduction angles.
    Returns (knee_angle_rmse, hip_abduction_angle_rmse).
    """
    name_to_idx = {name: i for i, name in enumerate(kp_order)}
    needed = ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE", 
              "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"]
    if not all(n in name_to_idx for n in needed):
        return np.nan, np.nan

    lh_idx = name_to_idx["LEFT_HIP"]
    lk_idx = name_to_idx["LEFT_KNEE"]
    la_idx = name_to_idx["LEFT_ANKLE"]
    rh_idx = name_to_idx["RIGHT_HIP"]
    rk_idx = name_to_idx["RIGHT_KNEE"]
    ra_idx = name_to_idx["RIGHT_ANKLE"]

    left_knee_diff_sq = []
    right_knee_diff_sq = []
    left_hip_abd_diff_sq = []
    right_hip_abd_diff_sq = []

    F = correct_arr.shape[0]
    for f in range(F):
        c_l_hip, c_l_knee, c_l_ankle = correct_arr[f, lh_idx], correct_arr[f, lk_idx], correct_arr[f, la_idx]
        p_l_hip, p_l_knee, p_l_ankle = patient_arr[f, lh_idx], patient_arr[f, lk_idx], patient_arr[f, la_idx]

        c_r_hip, c_r_knee, c_r_ankle = correct_arr[f, rh_idx], correct_arr[f, rk_idx], correct_arr[f, ra_idx]
        p_r_hip, p_r_knee, p_r_ankle = patient_arr[f, rh_idx], patient_arr[f, rk_idx], patient_arr[f, ra_idx]

        # Knee angles
        c_l_knee_angle = compute_knee_angle(c_l_hip, c_l_knee, c_l_ankle)
        p_l_knee_angle = compute_knee_angle(p_l_hip, p_l_knee, p_l_ankle)
        c_r_knee_angle = compute_knee_angle(c_r_hip, c_r_knee, c_r_ankle)
        p_r_knee_angle = compute_knee_angle(p_r_hip, p_r_knee, p_r_ankle)

        left_knee_diff_sq.append((c_l_knee_angle - p_l_knee_angle) ** 2)
        right_knee_diff_sq.append((c_r_knee_angle - p_r_knee_angle) ** 2)

        # Hip abduction angles
        c_l_hip_abd = compute_hip_abduction_angle(c_l_hip, c_l_knee)
        p_l_hip_abd = compute_hip_abduction_angle(p_l_hip, p_l_knee)
        c_r_hip_abd = compute_hip_abduction_angle(c_r_hip, c_r_knee)
        p_r_hip_abd = compute_hip_abduction_angle(p_r_hip, p_r_knee)

        left_hip_abd_diff_sq.append((c_l_hip_abd - p_l_hip_abd) ** 2)
        right_hip_abd_diff_sq.append((c_r_hip_abd - p_r_hip_abd) ** 2)

    # Mean of squared differences
    left_knee_rmse = np.sqrt(np.mean(left_knee_diff_sq))
    right_knee_rmse = np.sqrt(np.mean(right_knee_diff_sq))
    left_hip_abd_rmse = np.sqrt(np.mean(left_hip_abd_diff_sq))
    right_hip_abd_rmse = np.sqrt(np.mean(right_hip_abd_diff_sq))

    knee_angle_rmse = 0.5 * (left_knee_rmse + right_knee_rmse)
    hip_abd_rmse = 0.5 * (left_hip_abd_rmse + right_hip_abd_rmse)

    return knee_angle_rmse, hip_abd_rmse

def upload_rsme_to_snowflake(coordinate_rmses, 
                                  keypoints_of_interest, 
                                  knee_rmse, hip_abd_rmse, 
                                  table_name="stg_rmse_results"):

    data = {
        "keypoint_name": keypoints_of_interest + ["KNEE_ANGLE", "HIP_ABDUCTION_ANGLE"],
        "RMSE": coordinate_rmses + [knee_rmse, hip_abd_rmse]
    }
    df = pd.DataFrame(data)

    conn = connect_snowflake()
    cursor = conn.cursor()

    # Load environment variables
    load_dotenv()

    database = os.getenv("SNOWFLAKE_DATABASE")
    schema = os.getenv("SNOWFLAKE_SCHEMA")

    # Create table if not exists
    qualified_table_name = f"{database}.{schema}.{table_name}"
    create_table_query = f'''
    CREATE TABLE IF NOT EXISTS {qualified_table_name} (
        keypoint_name STRING,
        RMSE FLOAT
    );
    '''
    cursor.execute(create_table_query)

    # Insert values into Snowflake
    insert_query = f"""
    INSERT INTO {qualified_table_name} (keypoint_name, RMSE) 
    VALUES (%s, %s)
    """
    for _, row in df.iterrows():
        cursor.execute(insert_query, (row['keypoint_name'], row['RMSE']))

    conn.commit()
    cursor.close()
    conn.close()
    print(f"[result] RMSE results imported to Snowflake table {qualified_table_name}")

    # Run DBT transformation after data upload
    try:
        print("[info] Running DBT transformations...")
        subprocess.run(
            ["dbt", "run"],
            check=True,
            cwd="./dbt_project"
        )
        print("[success] DBT run completed.")
    except subprocess.CalledProcessError as e:
        print(f"[error] DBT run failed: {e}")

