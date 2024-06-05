import sys

sys.path.append('../')
from utils import get_box_centre, measure_distance

#create a class for the ball possession
class BallPossession():

    def __init__(self):
        #set the maximum distance that the ball can be from the player without losing indication
        self.max_distance_from_player = 60

    #function to assign ball possession
    def assign_ball_possession(self, players, ball_bbox):
        #retrieve ball position
        ball_position = get_box_centre(ball_bbox)

        #initial values of min distance and assigned player to be overwritten
        minimum_distance  = 8000
        assigned_player=-1
        #loop over each player within the frame
        for player_id, player in players.items():
            player_bbox = player['bbox']

            #ditance between left x and bottom y coords
            distance_left = measure_distance((player_bbox[0],player_bbox[-1]),ball_position)
            #distance between right x and bottom y coords
            distance_right = measure_distance((player_bbox[2],player_bbox[-1]),ball_position)
            #mimumum distance of the two
            distance = min(distance_left,distance_right)

            #check distance, assign possession to a player if within range
            if distance < self.max_distance_from_player:
                if distance < minimum_distance:
                    mimumum_distance = distance
                    assigned_player = player_id

        return assigned_player

