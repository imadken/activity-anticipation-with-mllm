import cv2
import os

# Directory containing your frames
frames_dir = "data/epickitchen"  # Replace with your frames directory path
output_video = "output_video.mp4"    # Name of the output video file



# Get all frame files
frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

# Read first frame to get dimensions
first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
height, width, layers = first_frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use 'XVID'
fps = 5  # Frames per second - adjust as needed
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Write each frame to video
for frame_file in frame_files:
    frame_path = os.path.join(frames_dir, frame_file)
    frame = cv2.imread(frame_path)
    
    if frame is None:
        print(f"Failed to load frame: {frame_file}")
        continue

    # cv2.imshow('Frame', frame)
    video_writer.write(frame)
    print(f"Added frame: {frame_file}")

# Release the video writer
video_writer.release()
print(f"Video saved as {output_video}")