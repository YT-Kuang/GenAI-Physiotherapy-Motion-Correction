# PhysioPro - GenAI for Physiotherapy Motion Correction

**PhysioPro** is an AI-powered physiotherapy motion analysis system that provides real-time movement correction and personalized rehabilitation feedback for at-home recovery, aiming to enhance patient engagement and improve recovery outcomes.
This project focuses on the **Fire Hydrant** exercise, utilizing video datasets from [Cornell University](https://www.cornell.edu/video/fire-hydrant-exercise) and [YouTube](https://www.youtube.com/watch?v=La3xYT8MGks) for model training and validation.

## What It Does

PhysioPro analyzes patient movements during rehabilitation exercises and provides:

- Real-time motion correction with visual overlays
- Personalized feedback using AI-generated instructions
- Movement quality assessment to track recovery progress
- Adaptive guidance tailored to individual patient needs

## Key Features

- **AI Motion Analysis**: Detects rehabilitation errors and movement patterns
- **Real-Time Feedback**: Provides immediate corrective guidance during exercises
- **Personalized Instructions**: Generates customized rehabilitation plans
- **Progress Tracking**: Monitors recovery outcomes and engagement metrics

## Technology Stack

- **Programming Language**: Python, SQL
- **AI**: OpenAI API, Snowflake Cortex
- **Data Preprocessing**: OpenCV, MediaPipe, Dynamic Time Warping (DTW)
- **Data Pipeline**: Airflow, dbt
- **Storage**: AWS S3, Snowflake
- **Frontend**: Streamlit

## Workflow Overview

Our AI-powered physiotherapy analysis follows a comprehensive 5-step pipeline:

<img width="800" alt="7374_project_architecture" src="https://github.com/user-attachments/assets/98652a53-3b53-42e4-adb8-65509c163cfb" />

**1. Video Upload & Storage**

- Patients upload movement videos through Streamlit interface
- Videos are securely stored in AWS S3 for processing
- Real-time feedback and motion overlays are displayed to users

**2. Motion Analysis**

- MediaPipe extracts keypoints from patient videos frame-by-frame
- Keypoint data is converted to JSON format and stored in S3
- Creates detailed movement pattern analysis for comparison

**3. Data Preprocessing & Alignment**

- Retrieves correct movement keypoint data from Snowflake
- Performs spatial alignment between patient and reference movements
- Uses Dynamic Time Warping (DTW) to find optimal alignment path with minimum differences

**4. Movement Comparison & Visualization**

- OpenCV generates visual overlays comparing patient vs. correct movements
- Calculates RMSE metrics to quantify movement accuracy
- Provides quantitative assessment of rehabilitation performance

**5. Personalized Feedback Generation**

- Snowflake Cortex** and OpenAI API analyze overlay images and RMSE metrics using Chain-of-Thought (CoT) prompting
- Generates personalized, text-based rehabilitation feedback
- Delivers actionable insights for movement improvement

## Problem Statement

Traditional physiotherapy rehabilitation faces critical challenges:

- **Limited Access**: Up to 50% of patients drop out due to transportation barriers, scheduling conflicts, and healthcare staff shortages
- **High Costs**: In-person physiotherapy sessions are financially prohibitive for many patients
- **Poor Adherence**: Patients struggle with correct movement execution during at-home rehabilitation without immediate guidance
- **Generic Care**: One-size-fits-all exercises don't account for individual progress or specific conditions
- **Risk of Reinjury**: Incorrect movements without real-time correction can delay recovery or worsen injuries

## Our Solution

**PhysioPro** bridges this gap by bringing AI-powered physiotherapy guidance directly to patients' homes, providing personalized, real-time motion correction and feedback.

## Impact

- Improved patient engagement through personalized feedback
- Reduced rehabilitation costs and accessibility barriers
- Enhanced recovery outcomes with real-time motion correction
- Scalable solution for underserved healthcare areas

## Getting Started

Follow these steps to set up and run the project:

1. **Create a Snowflake Account**  
   You'll need access to a Snowflake account to store and query motion data.

2. **Clone the repository** and **Set up a virtual environment**
    ```bash
    git clone https://github.com/your-username/PhysioPro.git
    cd PhysioPro
    python -m venv venv
    source venv/bin/activate

3. **Install Dependencies**  
   Run the following command to install all required packages:  
   ```bash
   pip install -r requirements.txt

4. **Initialize DBT** 
    Make sure DBT is properly initialized and configured for your environment:
    ```bash
    dbt init
    dbt debug

5. **Run the Streamlit App**
    Launch the app and upload a patient video (e.g., `patient_video.mp4`):
    ```bash
    streamlit run streamlit_app.py

6. **Receive AI-Powered Feedback**
    The system will analyze the motion and provide real-time physiotherapy feedback.

## Results & Analysis

| Keypoint Name  | RMSE |
| ------------- | ------------- |
| LEFT_HIP  | 0.04442004859447480  |
| RIGHT_HIP  | 0.04442005231976510  |
| LEFT_KNEE  | 0.1529528647661210  |
| RIGHT_KNEE  | 0.10703086107969300  |
| LEFT_ANKLE  | 0.25655630230903600  |
| RIGHT_ANKLE  | 0.29119545221328700  |
| LEFT_HEEL  | 0.27717193961143500  |
| RIGHT_HEEL  | 0.3226935565471650  |
| LEFT_FOOT_INDEX  | 0.27292144298553500  |
| RIGHT_FOOT_INDEX  | 0.33592307567596400  |
| KNEE_ANGLE_RMSE  | 19.86235160709360  |
| HIP_ABDUCTION_ANGLE_RMSE  | 30.394218496441100  |

<table>
  <tr>
    <td>
       <img width="300" alt="report1" src="https://github.com/user-attachments/assets/49c661dd-7e0d-472b-a5fc-1905ea10c47f" />
    </td>
    <td>
       <img width="300" alt="report2" src="https://github.com/user-attachments/assets/2877462e-127d-494f-8d3f-40cf652f1ab4" />
    </td>
  </tr>
</table>

**Key Findings**:

- Highest Error Areas: Right foot movement showed significant deviations (RIGHT_FOOT_INDEX: 0.336, RIGHT_HEEL: 0.323, RIGHT_ANKLE: 0.291)
- Hip Abduction Issues: 30.39° RMSE indicating substantial hip movement deviation
- Knee Alignment Problems: 19.86° RMSE suggesting alignment issues during exercise execution
- Left Side Compensation: Elevated RMSE values in left ankle and heel indicating compensatory movement patterns

**AI-Generated Personalized Feedback**:

Our system provides step-by-step corrective recommendations:

1. Right Foot Stabilization: Focus on toe raises and lateral foot movements to improve stability
2. Hip Stabilization: Incorporate clamshell exercises, lateral leg raises, and glute bridges for better hip control
3. Knee Alignment: Implement wall sits and proper squatting techniques to correct alignment
4. Left Side Balance: Use single-leg stands and proprioceptive exercises to reduce compensation patterns
5. Overall Coordination: Develop balanced strength training for symmetrical muscle development

This analysis framework enables healthcare providers to deliver precision-based rehabilitation through quantitative movement assessment. Clinicians can identify specific deficiencies using objective RMSE measurements rather than subjective evaluations, allowing for targeted, evidence-based recommendations tailored to each patient's unique movement profile. The system facilitates real-time progress tracking and protocol adjustments, ultimately delivering personalized care plans that optimize recovery outcomes through scientific precision.

##  Future Optimizations
- Performance Enhancement: Optimize visual overlay rendering and RMSE calculations to reduce 58% computational overhead
- Adaptive Learning: Implement reinforcement learning for real-time personalized motion correction guidance
- Advanced Computing: Leverage spiking neural networks with neuromorphic computing for enhanced energy efficiency and temporal processing

## References
1. Pose Estimation (MediaPipe Pose) Dill, S., Rösch, A., Rohr, M., Güney, G., De Witte, L., Schwartz, E., & Hoog Antink, C. (2023). Accuracy Evaluation of 3D Pose
Estimation with MediaPipe Pose for Physical Exercises. Current Directions in Biomedical Engineering, 9(1), 563-566. [https://doi.org/10.1515/cdbme-2023-1141]
2. Timely alignment (Dynamic Time Warping, DTW) Switonski, A., Josinski, H., & Wojciechowski, K. (2019). Dynamic time warping in classification and selection
of motion capture data. Multidimensional Systems and Signal Processing, 30, 1437–1468. [https://doi.org/10.1007/s11045-018-0611-3]
3. LLM-based Feedback Feng, Yao, et al. "ChatPose: Chatting about 3D Human Pose. " Max Planck Institute for Intelligent Systems, ETH Zürich, Meshcapade,
Tsinghua University, 2024. [https://arxiv.org/pdf/2311.18836]
4. Kim, J.-W., Choi, J.-Y., Ha, E.-J., & Choi, J.-H. (2023). Human Pose Estimation Using MediaPipe Pose and Optimization Method Based on a Humanoid Model.
Applied Sciences, 13(4), 2700. [https://doi.org/10.3390/app13042700]
5. Zhou, F., & De la Torre, F. (2012). Generalized time warping for multi-modal alignment of human motion. 2012 IEEE Conference on Computer Vision and
Pattern Recognition (CVPR), 1282-1289. [https://doi.org/10.1109/CVPR.2012.6247812]
6. Tevet, G., Gordon, A., & Bermano, A. (2023). Human Motion Diffusion Model. arXiv preprint. [https://arxiv.org/pdf/2209.14916]
7. Stent, M. (n.d.). Dynamic Time Warping. Medium. [https://medium.com/@markstent/dynamic-time-warping-a8c5027defb6]
8. Dynamic Time Warping in Motion Data Analysis. Blog [https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/122904252]

