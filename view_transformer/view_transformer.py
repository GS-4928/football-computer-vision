import numpy as np
import cv2

class ViewTransformer():

    def __init__(self):
        #width of football pitch
        pitch_w = 68
        #length of segment of football pitch
        pitch_l = 23.32

        #provide the position of the points that map with this section of the pitch
        self.pixel_verticies = np.array([
            [110,1035],
            [265,275],
            [910,260],
            [1640,915]
        ])

        #provide the target corners of the size of the pitch that the trapezoid corresponds to
        self.target_verticies = np.array([
            [0,pitch_w],
            [0,0],
            [pitch_l,0],
            [pitch_l,pitch_w]
        ])

        #cast each set of corners as floats
        self.pixel_verticies = self.pixel_verticies.astype(np.float32)
        self.target_verticies = self.target_verticies.astype(np.float32)

        #transform the image into the desired measurements
        self.perpective_transformer = cv2.getPerspectiveTransform(self.pixel_verticies,self.target_verticies)

    #transform the points based on transformer
    def transform_point(self,point):
        p  = (int(point[0]),int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_verticies,p,False) >= 0
        if not is_inside:
            return None
        reshaped_point = point.reshape(-1,1,2).astype(np.float32)
        transform_point = cv2.perspectiveTransform(reshaped_point,self.perpective_transformer)

        return transform_point.reshape(-1,2)

    #add this transformed position to the tracks dictionary
    def add_transformed_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    transformed_position = self.transform_point(position)
                    if transformed_position is not None:
                        transformed_position = transformed_position.squeeze().tolist()
                    tracks[object][frame_num][track_id]['transformed_position'] = transformed_position
