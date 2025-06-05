# PhysioPro

**PhysioPro** is an end-to-end physiotherapy motion analysis pipeline designed to provide personalized rehabilitation feedback, enhance patient engagement, and improve recovery outcomes.

## 🚀 Getting Started

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

