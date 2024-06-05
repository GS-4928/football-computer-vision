import cv2
from utils import read_video, save_video, measure_distance
from trackers import Tracker
from team_assigner import TeamAssigner
from ball_possession import BallPossession
import numpy as np
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator

def main():
    #read in video from our input video folder
    video_frames = read_video('Input_Videos/08fd33_4.mp4')

    

    #initialise tracker
    tracker = Tracker('models/best.pt')

    #track our video frames
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    
    #calculate object positions
    tracker.add_position_to_tracks(tracks)
    
    #track the camera movement
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=True,
                                                                              stub_path ='stubs/camera_movement_stub.pkl')
    
    #adjust positions to account for camera movement
    camera_movement_estimator.adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    #transform view to reflect true dimensions of pitch
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    #interpolate the position of the ball for each missing frame
    tracks['ball'] = tracker.interpolate_ball_position(tracks['ball'])
    
    #estimate speed and distance travelled
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    #assign players to teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_colour(video_frames[0],
                                     tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_colour'] = team_assigner.team_colours[team]
            
    #assign ball possession
    ball_possession = BallPossession()
    team_possession = []

    for frame_num, player_track in enumerate(tracks['players']):
        #pull out ball bounding box
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        #see whcih player is closest to the ball
        assigned_player = ball_possession.assign_ball_possession(player_track, ball_bbox)

        #if the assigned player value has changed, update the possession of the ball
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_possession.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_possession.append(len(team_possession))

    team_possession = np.array(team_possession)

    #draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames,tracks, team_possession)

    #draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,
                                                                        camera_movement_per_frame)
    
    #draw speed and distance
    speed_and_distance_estimator.draw_speed_distance_annotations(output_video_frames,tracks)
    #save our video
    save_video(output_video_frames, 'Output_Videos/08fd33_4_output_final.avi')

if __name__ == '__main__':
    main()