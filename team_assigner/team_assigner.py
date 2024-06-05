from sklearn.cluster import KMeans
class TeamAssigner:

    def __init__(self):
        #set up dictionary of team colours
        self.team_colours = {}
        #set up dictionary of which player belongs to which team
        self.player_team_dict = {}

    #function to produce our clustering model for each image
    def get_clustering_model(self, image):
        #reshape the image into 2d array
        image_2d = image.reshape(-1,3)

        #instantiate KMeans model with 2 clusters: one for background and one for player
        kmeans = KMeans(n_clusters=2,random_state=0,n_init=10,init='k-means++')
        #fit our model to each image
        kmeans.fit(image_2d)
        #return our model
        return kmeans

    #function deciding on the colour of each player
    def get_player_colour(self, frame, bbox):
        #crop the image around each player
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        #get top half of the image
        image_top_half = image[0:int(image.shape[0]/2)]

        #provide clustering model
        kmeans = self.get_clustering_model(image_top_half)

        #pull out cluster lables for each pixel
        cluster_labels = kmeans.labels_

        #reshape to image shape
        clustered_image = cluster_labels.reshape(image_top_half.shape[0],image_top_half.shape[1])

        #pull out corner clusters
        corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters),key=corner_clusters.count)

        #define the player cluster
        player_cluster = 1-non_player_cluster

        #pull out the colour of the player cluster in RGB format
        player_colour = kmeans.cluster_centers_[player_cluster]

        return player_colour
    
    #function to assign team colours based on player colours
    def assign_team_colour(self, frame, player_detections):
        #set up list of player colours to be added to
        player_colours = []

        for _, player_detection in player_detections.items():

            #pull out the bounding box of each player
            bbox = player_detection['bbox']

            #use the function created within the notebook
            player_colour = self.get_player_colour(frame, bbox)

            #add player colour to the list
            player_colours.append(player_colour)

        #instantiate KMeans model to split player colours into 2 clusters
        kmeans = KMeans(n_clusters=2,random_state=0,n_init=10,init='k-means++')
        #fit the model to our player colours
        kmeans.fit(player_colours)

        #save this kmeans model to refer back to it later
        self.kmeans = kmeans

        #assign team colours within initial dictionary to cluster colours
        self.team_colours[0] = kmeans.cluster_centers_[0]
        self.team_colours[1] = kmeans.cluster_centers_[1]

    
    #function to assign players to each team
    def get_player_team(self, frame, player_bbox, player_id):

        #if the player is already within the dictionary, return the corresponding team
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        #if not in dictionary, get the player colour
        player_colour = self.get_player_colour(frame, player_bbox)

        #set the team id to the prediction of the kmeans model
        team_id = self.kmeans.predict(player_colour.reshape(1,-1))[0]
        
        #hardcode goalkeeper of white team to be white
        if player_id == 91:
            team_id = 1

        #save this team id and player id combo to the player team dictionary
        self.player_team_dict[player_id] = team_id

        return team_id