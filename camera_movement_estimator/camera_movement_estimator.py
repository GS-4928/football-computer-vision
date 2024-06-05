import pickle
import cv2
import numpy as np
import sys
sys.path.append('../')
from utils import measure_distance, measure_xy_distance
import os

class CameraMovementEstimator:
    
    def __init__(self,frame):

        #minimum camera movement
        self.minimum_distance = 5
        #specify lk params
        self.lk_params = dict(
            winSize = (15,15),
            maxLevel = 2,
            criteria = (cv2.TermCriteria_EPS | cv2.TermCriteria_COUNT, 10, 0.03)
        )

        first_frame_greyscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #choosing features from the top and bottom of the frame that shouldn't be obstructed by movement
        mask_features = np.zeros_like(first_frame_greyscale)
        #slicing out so we have the top 20 rows of pixels and the bottom banners
        mask_features[:,0:20] = 1
        mask_features[:,900:1050] = 1

        #setting up dictionary to use for feature extraction
        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance = 3,
            blockSize = 7,
            mask = mask_features
        )
    
    #adjust the positions of objects accounting for camera movement
    def adjust_positions_to_tracks(self,tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = position[0] - camera_movement[0], position[1] - camera_movement[1]
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted


    def get_camera_movement(self, frames, read_from_stub=False,stub_path=None):
        #read the stub when present to cut down processing time
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)
            
        #set up camera movement array
        camera_movement = [[0,0]]*len(frames)

        #greyscale the frames that have been passed through
        old_grey = cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY)
        #extract relevant features from the passed frames, the ** before features allows for dictionary
        #to be expanded into features
        old_features = cv2.goodFeaturesToTrack(old_grey,**self.features)

        for frame_num in range(1,len(frames)):
            frame_grey = cv2.cvtColor(frames[frame_num],cv2.COLOR_BGR2GRAY)
            #pull out new features, status and error are given as wildcards, not needed
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_grey,
                                                          frame_grey,
                                                          old_features,
                                                          None,
                                                          **self.lk_params)
            
            #measure maximum distance between features
            max_distance = 0
            camera_movement_x, camera_movement_y = 0,0

            #unspool old and new features
            for _, (new,old) in enumerate(zip(new_features, old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                #measure the distance between old and new features, updating max if needed
                distance = measure_distance(new_features_point,old_features_point)
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_features_point,
                                                                               new_features_point)
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x,camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_grey,**self.features)

            old_grey = frame_grey.copy()
        
        #store stubs after first run through
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement,f)

        return camera_movement
    
    def draw_camera_movement(self,frames,camera_movement_per_frame):
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            overlay = frame.copy()

            #drawing housing for camera movement
            cv2.rectangle(overlay,(50,50),(600,150),(255,255,255),cv2.FILLED)
            alpha = 0.6
            cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame,
                                f'Camera - X Movement: {x_movement:.2f}',
                                (75,90),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0,0,0),
                                3)
            
            frame = cv2.putText(frame,
                                f'Camera - Y Movement: {y_movement:.2f}',
                                (75,140),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0,0,0),
                                3)
            
            output_frames.append(frame)

        return output_frames