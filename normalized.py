import numpy as np
import mediapipe as mp
from fastdtw import fastdtw
import tempfile
import boto3
import json
import time
from scipy.spatial.distance import euclidean

s3_client = boto3.client('s3')

def align_skeleton_to_standard(keypoints_data):
    """
    Align the keypoints data to a "standard human coordinate system":
      - X-axis: from the midpoint of the hips (hip_mid) to the midpoint of the shoulders (shoulder_mid) => +X direction
      - Y-axis: from left shoulder to right shoulder => +Y
      - Forward: -Z-axis
    
    Returns a new dictionary with the same structure as the input, but with coordinates shifted and rotated.
    """
    # Custom standard coordinate system's three basis vectors
    standard_x = np.array([-1, 0, 0], dtype=np.float32)  # You changed this manually to -X
    standard_y = np.array([0, 1, 0], dtype=np.float32)
    standard_z = np.array([0, 0, -1], dtype=np.float32)

    aligned_data = {}

    for frame_idx, frame_dict in keypoints_data.items():
        
        needed_keys = ["LEFT_HIP", "RIGHT_HIP", "LEFT_SHOULDER", "RIGHT_SHOULDER"]
        if not all(k in frame_dict for k in needed_keys):
            aligned_data[frame_idx] = frame_dict
            continue
        
        left_hip = frame_dict["LEFT_HIP"]
        right_hip = frame_dict["RIGHT_HIP"]
        hip_mid = (left_hip + right_hip) / 2.0

        left_shoulder = frame_dict["LEFT_SHOULDER"]
        right_shoulder = frame_dict["RIGHT_SHOULDER"]
        shoulder_mid = (left_shoulder + right_shoulder) / 2.0

        # Local coordinates
        local_x = shoulder_mid - hip_mid  # Upward
        if np.linalg.norm(local_x) < 1e-6:
            aligned_data[frame_idx] = frame_dict
            continue
        local_x /= np.linalg.norm(local_x)

        local_y = right_shoulder - left_shoulder  # Left -> Right
        if np.linalg.norm(local_y) < 1e-6:
            aligned_data[frame_idx] = frame_dict
            continue
        local_y /= np.linalg.norm(local_y)

        local_z = np.cross(local_x, local_y)  # Backward
        if np.linalg.norm(local_z) < 1e-6:
            aligned_data[frame_idx] = frame_dict
            continue
        local_z /= np.linalg.norm(local_z)

        M_local = np.stack([local_x, local_y, local_z], axis=1)
        M_std = np.stack([standard_x, standard_y, standard_z], axis=1)
        R = M_std @ M_local.T

        new_frame_dict = {}
        for k_name, coord in frame_dict.items():
            # Shift to hip_mid and then rotate
            shifted = coord - hip_mid
            rotated = R @ shifted
            new_frame_dict[k_name] = rotated
        
        aligned_data[frame_idx] = new_frame_dict

    return aligned_data

def compute_local_axes_for_frame(frame_dict):
    """
    From a single frame's data, compute the 3x3 local coordinate system matrix (x, y, z) column vectors.
    Assumptions:
      x: Midpoint of hips -> Midpoint of shoulders
      y: Left shoulder -> Right shoulder
      z: x cross y
    """
    needed_keys = ["LEFT_HIP", "RIGHT_HIP", "LEFT_SHOULDER", "RIGHT_SHOULDER"]
    if not all(k in frame_dict for k in needed_keys):
        return None
    
    left_hip = frame_dict["LEFT_HIP"]
    right_hip = frame_dict["RIGHT_HIP"]
    hip_mid = (left_hip + right_hip) / 2.0
    
    left_shoulder = frame_dict["LEFT_SHOULDER"]
    right_shoulder = frame_dict["RIGHT_SHOULDER"]
    shoulder_mid = (left_shoulder + right_shoulder) / 2.0

    local_x = shoulder_mid - hip_mid
    if np.linalg.norm(local_x) < 1e-6:
        return None
    local_x /= np.linalg.norm(local_x)

    local_y = right_shoulder - left_shoulder
    if np.linalg.norm(local_y) < 1e-6:
        return None
    local_y /= np.linalg.norm(local_y)

    local_z = np.cross(local_x, local_y)
    if np.linalg.norm(local_z) < 1e-6:
        return None
    local_z /= np.linalg.norm(local_z)

    M_local = np.stack([local_x, local_y, local_z], axis=1)
    return M_local

def single_global_rotation(aligned_correct, aligned_patient):
    """
    For two already aligned skeleton data (via align_skeleton_to_standard),
    select one frame (e.g., the one with the smallest frame_idx) and compute a rotation matrix R
    to align the "standard video" frame's coordinate system with the "patient video" frame's coordinate system.
    
    Returns the calculated rotation matrix R, which is used later in apply_global_rotation_to_dict.
    """
    cframe = min(aligned_correct.keys())
    pframe = min(aligned_patient.keys())

    if cframe not in aligned_correct or pframe not in aligned_patient:
        return np.eye(3, dtype=np.float32)

    M_c = compute_local_axes_for_frame(aligned_correct[cframe])
    M_p = compute_local_axes_for_frame(aligned_patient[pframe])
    if M_c is None or M_p is None:
        return np.eye(3, dtype=np.float32)

    # R = M_p * M_c^T
    R = M_p @ M_c.T
    return R

def apply_global_rotation_to_dict(keypoints_data, R):
    """
    Apply the same rotation matrix R to the entire keypoints_data (all frames).
    Note: This is the final rotation after the skeleton has already been translated into a "world coordinate system."
    """
    rotated_dict = {}
    for fidx, frame_dict in keypoints_data.items():
        new_frame = {}
        for k_name, coord in frame_dict.items():
            new_frame[k_name] = R @ coord  # Perform overall rotation in the unified coordinate system
        rotated_dict[fidx] = new_frame
    return rotated_dict

def dictionary_to_frame_array(keypoints_dict, keypoints_order=None):
    """
    Convert {frame_idx: {key_name: (x, y, z)...}, ...} into 
    a shape=(num_frames, num_keypoints, 3) array, where frame_idx is sorted.
    
    :param keypoints_order: If provided, the data will be strictly ordered based on this keypoint sequence.
    """

    if keypoints_order is None:
        # Retrieve the keypoint order from mediapipe Pose
        mp_pose = mp.solutions.pose
        keypoints_order = [landmark.name for landmark in mp_pose.PoseLandmark]
    
    frames_sorted = sorted(keypoints_dict.keys())
    
    array_list = []
    for fidx in frames_sorted:
        kp_coords = []
        frame_data = keypoints_dict[fidx]
        for kname in keypoints_order:
            if kname in frame_data:
                kp_coords.append(frame_data[kname])
            else:
                kp_coords.append(np.array([np.nan, np.nan, np.nan], dtype=np.float32))
        array_list.append(kp_coords)
    
    arr = np.array(array_list)  # shape=(num_frames, num_keypoints, 3)
    return arr, frames_sorted, keypoints_order

def dtw_time_warping(source_array, target_array):
    """
    Use fastdtw to align `source_array` to `target_array` in time.
    Returns `aligned_source` with the same number of frames as `target_array`.
    
    Each array has shape (F, K, 3).
    """
    sF, sK, s3 = source_array.shape
    tF, tK, t3 = target_array.shape

    # Flatten each frame (K*3) so we can apply DTW on 1D vectors per frame
    source_flat = source_array.reshape(sF, sK * s3)
    target_flat = target_array.reshape(tF, tK * t3)

    # Perform DTW
    _, path = fastdtw(source_flat, target_flat, dist=euclidean)

    # Build new array matching target's length
    aligned_source = np.zeros((tF, sK, s3), dtype=np.float32)
    fill_count = {}

    for (si, ti) in path:
        aligned_source[ti] = source_array[si]
        fill_count[ti] = fill_count.get(ti, 0) + 1

    # Fill any frames that did not get assigned (simple approach: copy nearest neighbor)
    for i in range(tF):
        if i not in fill_count:
            # fallback: copy from i-1 or i+1
            if i > 0 and (i - 1) in fill_count:
                aligned_source[i] = aligned_source[i - 1]
            else:
                # look forward
                j = i + 1
                while j < tF and j not in fill_count:
                    j += 1
                if j < tF:
                    aligned_source[i] = aligned_source[j]

    return aligned_source

def normalized_process(correct_video_keypoints, 
                       patient_video_keypoints,
                       bucket_name,
                       skeleton_data_s3_path,
                       keypoints_order_s3_path):

    start_time = time.time()

    # 1) Align each skeleton to a standard local frame
    aligned_correct = align_skeleton_to_standard(correct_video_keypoints)
    aligned_patient = align_skeleton_to_standard(patient_video_keypoints)

    # 2) Apply a single global rotation so that the "correct" orientation is closer to the patient's
    R_global = single_global_rotation(aligned_correct, aligned_patient)
    aligned_correct = apply_global_rotation_to_dict(aligned_correct, R_global)

    # 3) Convert each dictionary into a NumPy array
    correct_arr, correct_frames, kp_order = dictionary_to_frame_array(aligned_correct)
    patient_arr, patient_frames, _ = dictionary_to_frame_array(aligned_patient, kp_order)

    # 4) Time warp (DTW) the patient array to match the correct array length
    patient_aligned_arr = dtw_time_warping(patient_arr, correct_arr)

    # Step 5: Save skeleton arrays as JSON
    keypoints_data = {
        "correct": correct_arr.tolist(),
        "patient_aligned": patient_aligned_arr.tolist()
    }

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as temp_skeleton:
        json.dump(keypoints_data, temp_skeleton)
        temp_skeleton.flush()
        s3_client.upload_file(temp_skeleton.name, bucket_name, skeleton_data_s3_path)
    print(f"[normalized] Skeleton keypoint arrays uploaded to s3://{bucket_name}/{skeleton_data_s3_path}")

    # Step 6: Save kp_order as a Python-style list string
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as temp_kp:
        temp_kp.write(str(kp_order))  # this writes it like a Python list, not JSON
        temp_kp.flush()
        s3_client.upload_file(temp_kp.name, bucket_name, keypoints_order_s3_path)
    print(f"[normalized] Keypoint order uploaded to s3://{bucket_name}/{keypoints_order_s3_path}")

    end_time = time.time()
    print(f"[normalized] Total normalization time: {end_time - start_time:.2f} seconds.")
    
    return correct_arr, patient_aligned_arr, kp_order