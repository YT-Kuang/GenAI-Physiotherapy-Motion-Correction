import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import time
from datetime import datetime
import concurrent.futures
# from sqlalchemy import create_engine
from utils import (connect_s3, 
                   connect_snowflake)
from main import main

# Load environment variables
load_dotenv()

PATIENT_VIDEO_BUCKET = os.getenv("PATIENT_VIDEO_BUCKET")

# # Snowflake configuration
# SNOWFLAKE_CONFIG = {
#     "user": os.getenv("SNOWFLAKE_USERNAME"),
#     "password": os.getenv("SNOWFLAKE_PASSWORD"),
#     "account": os.getenv("SNOWFLAKE_ACCOUNT"),
#     "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
#     "database": os.getenv("SNOWFLAKE_DATABASE"),
#     "schema": os.getenv("SNOWFLAKE_SCHEMA"),
# }

# # Create SQLAlchemy engine for Snowflake
# try:
#     engine = create_engine(
#     f"snowflake://{SNOWFLAKE_CONFIG['user']}:{SNOWFLAKE_CONFIG['password']}@{SNOWFLAKE_CONFIG['account']}/{SNOWFLAKE_CONFIG['database']}/{SNOWFLAKE_CONFIG['schema']}?warehouse={SNOWFLAKE_CONFIG['warehouse']}"
# )
# except Exception as e:
#     st.error(f"Failed to connect to Snowflake: {e}")
#     st.stop()

# Create uploads directory
UPLOAD_DIR = "./lower_extremity"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Streamlit UI setup
st.set_page_config(page_title="PhysioPro - Motion Feedback", layout="centered")
st.title("PhysioPro: AI Motion Correction & Feedback")

# Display current timestamp
now = datetime.now()
dt_string = now.strftime("%d %B, %Y %H:%M:%S")
st.write(f'Last update: {dt_string}')
st.markdown("""
Welcome to **PhysioPro**, your virtual motion correction assistant.
Please enter your **personal details**, upload your **motion video**, and you will receive AI-generated feedbacks.
""")

# Step 1: Collect user personal details
st.markdown("### 🧑‍⚕️ Enter Your Personal Information")

body_area = st.selectbox(
    "Select Target Area of Exercise:",
    options=["Lower Body (Legs, Hips, Knees)", "Upper Body (Shoulders, Arms)", "Core (Abdomen, Back)"]
)
# Internally map to standardized type for storage
area_map = {
    "Lower Body (Legs, Hips, Knees)": "LEG",
    "Upper Body (Shoulders, Arms)": "SHOULDERS",
    "Core (Abdomen, Back)": "CORE"
}
exercise_type = area_map[body_area]
# exercise_type = re.sub(r'\W+', '_', exercise_type.upper()) 
age = st.number_input("Age (years):", min_value=10, max_value=120, value=30)
weight = st.number_input("Weight (kg):", min_value=20, max_value=200, value=70)
height = st.number_input("Height (cm):", min_value=50, max_value=250, value=170)

# Display user details
# st.write(f"**Your Details:** Age: {age} years, Weight: {weight} kg, Height: {height} cm")
st.write(f"**Your Details:** Age: {age} years, Weight: {weight} kg, Height: {height} cm, Exercise: {exercise_type}")


# Step 2: Upload video
st.markdown("### 📤 Upload Your Motion Video")
video_file = st.file_uploader("Upload video (MP4/MOV)", type=["mp4", "mov"])

# Step 3 & 4: Show RMSE results, checkboxes, and LLM feedback after video upload
if video_file:
    st.video(video_file)
    try:
        # # Generate timestamp-based filename
        # # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"patient_video.mp4"
        video_path = os.path.join(UPLOAD_DIR, video_filename)
        
        # # Save video
        with open(video_path, "wb") as f:
            f.write(video_file.read())
        # st.success(f"Video uploaded and saved locally at {video_path}!")
        s3_client = connect_s3()
        s3_video_path = f"lower_extremity/{video_filename}"
        s3_client.upload_file(video_path, PATIENT_VIDEO_BUCKET, s3_video_path)
        st.success(f"Video uploaded to s3://{PATIENT_VIDEO_BUCKET}/{video_path}")
        os.remove(video_path)  # Cleanup

    except Exception as e:
        st.error(f"Failed to save video to S3: {e}")

    with st.spinner("Analyzing motion, please wait... this may take over a minute."):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(main, age, weight, height, exercise_type)
            llm_prompt, physio_feedback_json_str = future.result()

   # Fetch RMSE data
    try:
        conn = connect_snowflake()
        cur = conn.cursor()
        query = f"""
        SELECT * 
        FROM RMSE_RESULTS_0603
        WHERE TYPE = '{exercise_type}'
        """
        cur.execute(query)
        df = pd.DataFrame.from_records(
            cur.fetchall(), columns=["keypoint_name", "rmse", "TYPE"]
        )
        cur.close()
        conn.close()
        # df = pd.read_sql(query, engine)

        if df.empty:
           st.warning(f"No RMSE data found in RMSE_RESULTS_0603 table.")
    except Exception as e:
        st.error(f"Error fetching RMSE data: {e}")
        df = pd.DataFrame()

    # RMSE Results with Checkbox
    st.markdown("### 📊 Motion Analysis")
    with st.expander("**RMSE Values for Keypoints**"):
        if not df.empty:
            st.dataframe(df.rename(columns={"keypoint_name": "Keypoint", "rmse": "RMSE", "TYPE": "Exercise_Type"}))
        else:
            st.warning("No RMSE data to display.")

    # show_rmse = st.checkbox("RMSE Values for Keypoints fetched from Snowflake:")
    # if show_rmse and not df.empty:
    #     st.write("**RMSE Values for Keypoints:**")
    #     st.dataframe(df.rename(columns={"keypoint_name": "Keypoint", "rmse": "RMSE", "TYPE": "Exercise_Type"}))
    # elif show_rmse and df.empty:
    #     st.warning("No RMSE data to display.")

    # LLM Feedback with Prompt Checkbox
    st.markdown("### 🧠 AI Feedback")
    if not df.empty:
        with st.expander("**Generated LLM Prompt:**"):
            st.code(llm_prompt)

        # # Checkbox to show LLM prompt
        # show_prompt = st.checkbox("Show LLM Prompt")
        # if show_prompt:
        #     st.write("**Generated LLM Prompt:**")
        #     st.code(llm_prompt)

        st.write("**Generated Motion Feedback:**")
        st.success(physio_feedback_json_str)
    else:
        st.warning("No feedback available due to missing RMSE data.")
else:
    st.info("Please upload a video to view motion analysis and feedback.")

# Reset video upload time on new upload
if video_file and ('last_video' not in st.session_state or st.session_state.last_video != video_file.name):
    st.session_state.video_upload_time = time.time()
    st.session_state.last_video = video_file.name