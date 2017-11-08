from moviepy.editor import VideoFileClip
from IPython.display import HTML
import numpy as np
import cv2
from pipeline import ProcessImages

Output_video = 'test_output1.mp4'
# Input_video = 'project_video.mp4'
Input_video = 'test_video.mp4'

clip1 = VideoFileClip(Input_video)
process_images = ProcessImages()
video_clip = clip1.fl_image(process_images.run_pipeline)
video_clip.write_videofile(Output_video, audio=False)
