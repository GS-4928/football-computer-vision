import cv2

#define a function to read in our video files and add each frame to a list
def read_video(video_path):
    #capture the video using opencv
    cap = cv2.VideoCapture(video_path)
    #empty list for frames to be added to
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        #add frames
        frames.append(frame)
    #return frames
    return frames

def save_video(output_video_frames,output_video_path):
    #define an output format
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path,fourcc,24,
                          (output_video_frames[0].shape[1],output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()