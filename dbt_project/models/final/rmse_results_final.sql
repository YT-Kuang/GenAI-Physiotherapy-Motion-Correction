SELECT
    keypoint_name,
    RMSE,
    CASE
        WHEN keypoint_name IN (
            'LEFT_HIP', 'RIGHT_HIP',
            'LEFT_KNEE', 'RIGHT_KNEE',
            'LEFT_ANKLE', 'RIGHT_ANKLE',
            'LEFT_HEEL', 'RIGHT_HEEL',
            'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
        ) THEN 'lower_body'
        WHEN keypoint_name IN (
            'LEFT_SHOULDER', 'RIGHT_SHOULDER',
            'LEFT_ELBOW', 'RIGHT_ELBOW',
            'LEFT_WRIST', 'RIGHT_WRIST', 
            'LEFT_PINKY', 'RIGHT_PINKY',
            'LEFT_INDEX', 'RIGHT_INDEX',
            'LEFT_THUMB', 'RIGHT_THUMB'
        ) THEN 'upper_body'
        WHEN keypoint_name IN (
            'NOSE', 
            'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER',
            'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER', 
            'LEFT_EAR', 'RIGHT_EAR', 
            'MOUTH_LEFT', 'MOUTH_RIGHT'
        ) THEN 'head'
        WHEN keypoint_name IN (
            'KNEE_ANGLE', 'HIP_ABDUCTION_ANGLE'
        ) THEN 'metric'
        ELSE 'unknown'
    END AS type
FROM {{ ref('stg_rmse_results') }}
