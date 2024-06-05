#import for model
from ultralytics import YOLO
#import for tracking purposes
import supervision as sv
#import pickle for saving
import pickle
#import os to check the path
import os
import sys
import cv2
import numpy as np
import pandas as pd


#move back up one directory to expose the utils folder for importing centre and width functions
sys.path.append('../')
from utils import get_box_centre, get_box_width, get_foot_position

#create new tracker class
class Tracker:
    def __init__(self, model_path):
        #model set up on initiation
        self.model = YOLO(model_path)
        #tracker set up on instantiation
        self.tracker = sv.ByteTrack()
    #add position to tracks
    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_box_centre(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position
    #employ interpolation to fill in missing ball positions
    def interpolate_ball_position(self, ball_positions):
        #converting ball_positions into a pandas data frame
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_positions = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])

        #fill in missing values with linear interpolation
        df_positions = df_positions.interpolate()
        #for edge cases, back fill the missing values with the next available value
        df_positions = df_positions.bfill()

        ball_positions = [{1:{'bbox':x}}for x in df_positions.to_numpy().tolist()]

        return ball_positions

    #detect frames from our video files
    def detect_frames(self, frames):
        #set a batch size, avoid memory problems
        batch_size = 20
        #empty list to add into
        detections = []
        #loop over length of frames in this batch size
        for i in range(0,len(frames),batch_size):
            #predicting from the model, will be treating goalkeepers as normal players for this
            #so can't directly track yet as we want to overwrite or intial tracking data to remove
            #goalkeepers
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            #append to empty list
            detections += detections_batch
        #return the detections
        return detections
    #function to draw an ellipse around the players
    def draw_ellipse(self, frame, bbox, colour, track_id=None):
        #set the position of the box
        y2 = int(bbox[3])

        #define centre and width, denote y-centre as wildcard to cover parsing issue
        x_centre, _ = get_box_centre(bbox)
        width = get_box_width(bbox)

        #draw an ellipse around the players to mark them
        cv2.ellipse(
            frame,
            center=(x_centre,y2),
            axes=(int(width),int(0.35*width)),
            angle=0.0,
            startAngle=-20,
            endAngle=240,
            color=colour,
            thickness=3,
            lineType=cv2.LINE_4
        )
        
        #draw a shape for the tracking number of each player to be displayed
        #set height and width of rectangle
        rect_height = 20
        rect_width = 35
        #set coordinates for rectangle to sit at
        x1_rect = x_centre - rect_width//2 
        x2_rect = x_centre + rect_width//2
        y1_rect = (y2 - rect_height//2) + 30
        y2_rect = (y2 + rect_height//2) + 30

        #if a track_id is present, draw the rectangle to hold the track id
        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect),int(y1_rect)),
                (int(x2_rect),int(y2_rect)),
                colour,
                cv2.FILLED
            )
            
            #set up where the track id text should sit
            x1_text = x1_rect + 7
            if track_id > 99:
                x1_text -= 10

            #write in track id to the rectangle
            cv2.putText(frame,
                        f"{track_id}",
                        (int(x1_text),int(y2_rect-3)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.6,
                        color=(0,0,0),
                        thickness=2)
        
        return frame
    
    #function to draw a triangle above the ball
    def draw_triangle(self, frame, bbox, colour):
        y = int(bbox[1])
        x, _ = get_box_centre(bbox)

        triangle_points = np.array([[x,y],
                                   [x-10,y-15],
                                   [x+10,y-15]])
        
        #draw the triangle
        cv2.drawContours(frame,[triangle_points],0,colour,cv2.FILLED)
        #draw the outline of the triangle
        cv2.drawContours(frame,[triangle_points],0,(0,0,0),2)

        return frame
    #function to draw team possession stats
    def draw_team_possession(self, frame, frame_num, team_possession):
        #draw rectangle to fill with possession stats
        overlay = frame.copy()
        cv2.rectangle(overlay,
                     (1350,850),
                     (1900,970),
                     (255,255,255),
                     cv2.FILLED)
        #transparency level of rectangle
        alpha = 0.4
        #add to the frame
        cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)

        #list of frames until we reach current frame
        team_possession_frame = team_possession[:frame_num+1]
        #number of times each team had the ball
        team_1_num_frames = team_possession_frame[team_possession_frame == 1].shape[0]
        team_2_num_frames = team_possession_frame[team_possession_frame == 0].shape[0]
        team_1_possession = (team_1_num_frames / (team_1_num_frames + team_2_num_frames))*100
        team_2_possession = (team_2_num_frames / (team_1_num_frames + team_2_num_frames))*100

        #write this out within frame
        cv2.putText(frame,
                    f'Team 1 Possession: {team_1_possession:.2f}%',
                    (1400,900),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0,0,0),
                    thickness=3)
        cv2.putText(frame,
                    f'Team 2 Possession: {team_2_possession:.2f}%',
                    (1400,950),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0,0,0),
                    thickness=3)
        
        return frame
    #function to retrieve object tracks, either from existing stub or by running the code
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        #if we specify pathway that exists, load tracks and then return it without running rest of code
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        #set up dictionary to store tracking information
        tracks = {
            'players':[],
            'referees':[],
            'ball':[]
        }

        #look at respective classes within frame
        for frame_num, detection in enumerate(detections):
            class_names = detection.names
            class_names_inverse = {v:k for k,v in class_names.items()}
            print(class_names)

            #convert this detection to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            #convert goalkeepers to players
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if class_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_ind] = class_names_inverse['player']

            #Track objects
            detections_with_tracks = self.tracker.update_with_detections(detection_supervision)

            #add dictionary within each entry of the tracks dictionary, will contain track_id:bounding box
            #pair for each frame
            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})
            
            #loop through tracked frames, extract each class, track and bbox and add track id and bbox
            #if class information is correct
            for frame_detection in detections_with_tracks:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                track_id = frame_detection[4]

                if class_id == class_names_inverse['player']:
                    tracks['players'][frame_num][track_id] = {'bbox':bbox}

                if class_id == class_names_inverse['referee']:
                    tracks['referees'][frame_num][track_id] = {'bbox':bbox}
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]

                if class_id == class_names_inverse['ball']:
                    tracks['ball'][frame_num][1] = {'bbox':bbox}

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)
        #returning a list of dictionaries, frame number, tracking id and bounding box for player
        #referee and ball 
        return tracks
    #annotate the frames, using the drawing functions defined above
    def draw_annotations(self,video_frames,tracks, team_possession):
        #empty list to hold output frames once annotated
        output_video_frames = []
        #loop over index and frame in supplied video
        for frame_num, frame in enumerate(video_frames):
            #copy video frames, so we don't affect the original
            frame = frame.copy()

            #set up our dictionaries for players, ball, referees positions
            player_dict = tracks['players'][frame_num]
            ball_dict = tracks['ball'][frame_num]
            referee_dict = tracks['referees'][frame_num]

            #draw annotations around players
            for track_id, player in player_dict.items():
                colour = player.get('team_colour',(255,0,255))
                frame = self.draw_ellipse(frame, player['bbox'],colour,track_id)

                #if player is near the ball, denote this with triangle
                if player.get('has_ball',False):
                    frame = self.draw_triangle(frame, player['bbox'],(255,249,125))

            #draw annotations around referees, no track id as we only want this for players
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'],(0,255,255))

            #draw annotations for the ball
            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball['bbox'],(0,0,255))

            #draw team possession annotation
            frame = self.draw_team_possession(frame, frame_num, team_possession)

            #append resulting frames to output video and then return it
            output_video_frames.append(frame)

        return output_video_frames