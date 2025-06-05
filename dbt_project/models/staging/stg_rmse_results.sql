SELECT
    keypoint_name,
    RMSE
FROM {{ source('raw', 'stg_rmse_results') }}
