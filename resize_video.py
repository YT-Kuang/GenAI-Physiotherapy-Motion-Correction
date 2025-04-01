# -----------------------------------------
# Resized PT Exercise Videos
# -----------------------------------------

import cv2

def get_video_size(video_path):
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return None

    # Get width and height
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    
    return width, height

def resize_video(input_path, output_path, width, height):
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return
    
    # Get the original video properties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define the codec and create a VideoWriter object
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Break if no more frames
        
        # Resize the frame
        resized_frame = cv2.resize(frame, (width, height))
        
        # Write the resized frame to the output video
        out.write(resized_frame)
    
    # Release resources
    cap.release()
    out.release()
    print("Resizing complete. Output saved to:", output_path)

if __name__ == "__main__":
    patient_video_path = "./lower_extremity/patient_video.mp4"
    correct_video_path = "./lower_extremity/Fire_Hydrant.mp4"
    resized_video_path = "./lower_extremity/resized_correct_video.mp4"
    
    patient_width, patient_height = get_video_size(patient_video_path)
    
    resize_video(correct_video_path, resized_video_path, patient_width, patient_height)

    # Check the result
    correct_width, correct_height = get_video_size(resized_video_path)
    
    print(patient_width, patient_height)
    print(correct_width, correct_height)