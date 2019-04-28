import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib
import seaborn as sns
import planar



def homography_solve():
    """
    u = points in the video
    v = points on the map
    """
    v = np.array([[674, 318],
                  [546, 454],
                  [676, 580],
                  [800, 452]])

    u = np.array([[942, 301],
                  [453, 402],
                  [944, 517],
                  [1426, 402]])

    U = np.zeros((8, 2*u.shape[0]))
    for k in range(2*u.shape[1]):
        U[2*k,:] =   [u[k,0], u[k,1], 1, 0, 0, 0, -v[k, 0]*u[k, 0], -v[k, 0]*u[k, 1]]
        U[2*k+1,:] = [0, 0, 0, u[k,0], u[k,1], 1, -v[k, 1]*u[k, 0], -v[k, 1]*u[k, 1]]

    V = []
    for k in range(2*v.shape[1]):
        V.append(v[k,0])
        V.append(v[k,1])

    V = np.array(V)

    h = np.linalg.inv(U.T @ U) @ U.T @ V

    H = np.array([[h[0], h[1], h[2]],
                  [h[3], h[4], h[5]],
                  [h[6], h[7], 1]])
    return H

def homography_transform(u, H):
    u = np.append(u, 1)
    u_homogeneous = np.array(u).reshape(3,1)
    v_homogeneous = H @ u
    return [v_homogeneous[0]/v_homogeneous[2], v_homogeneous[1]/v_homogeneous[2]]

def create_gaussian_patch(size, sigma):
    patch = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            patch[i,j] = gaussian(i, j, int(size/2), int(size/2), sigma)
    patch /= np.sum(abs(patch))

    return patch

def gaussian(x, y, x0, y0, sigma):
    return np.exp(-((x-x0)**2 + (y-y0)**2)/(2*sigma**2))

def plot_heatmap(x_coord, y_coord, color):

    height = 892  #the height of the stadium image
    width = 1346  #the width of the stadium image

    y_coord = height - y_coord # to account for different convention on origin of field

    my_ax = sns.kdeplot(x_coord, y_coord, shade = "True", color = color, n_levels = 30)

    ## begin of image border lines ##
    lineCenter1 = matplotlib.patches.ConnectionPatch([width/4, 0], [width/3, height], "data", "data")
    lineCenter2 = matplotlib.patches.ConnectionPatch([3*width/4, 0], [2*width/3, height], "data", "data")
    lineLeft = matplotlib.patches.ConnectionPatch([width/2, 0], [width/3, height], "data", "data")
    lineRight = matplotlib.patches.ConnectionPatch([width/2, 0], [2*width/3, height], "data", "data")
    my_ax.plot([width/3,   width/4],   [0, height], '--', c='k')  # line center 1
    my_ax.plot([2*width/3, 3*width/4], [0, height], '--', c='k')  # line center 2
    my_ax.plot([width/3,   width/2],   [0, height], '--', c='k')  # line left
    my_ax.plot([2*width/3, width/2],   [0, height], '--', c='k')  # line right

    ## Drawing the white lines ##
    pitch = plt.Rectangle([0, 0], width = width, height = height, fill = False)
    leftPenalty = plt.Rectangle([0, 0.28*height], width = 0.12*width, height = 0.45*height, fill = False)
    rightPenalty = plt.Rectangle([0.88*width, 0.28*height], width = 0.12*width, height = 35.3/80*height, fill = False)
    midline = matplotlib.patches.ConnectionPatch([0.5*width, 0], [0.5*width, height], "data", "data")
    leftSixYard = plt.Rectangle([0, 0.4*height], width = 0.04*width, height = 0.2*height, fill = False)
    rightSixYard = plt.Rectangle([0.96*width, 0.4*height], width = 0.04*width, height = 0.2*height, fill = False)
    centreCircle = plt.Circle((0.5*width, 0.5*height), 0.09*width,color="black", fill = False)
    centreSpot = plt.Circle((0.5*width, 0.5*height), 0.005*width,color="black")
    leftPenaltySpot = plt.Circle((0.08*width, 0.5*height), 0.005*width, color="black")
    rightPenaltySpot = plt.Circle((0.92*width, 0.5*height), 0.005*width, color="black")

    elements = [pitch, leftPenalty, rightPenalty, midline, leftSixYard, rightSixYard, centreCircle, centreSpot, rightPenaltySpot, leftPenaltySpot]

    for i in elements:
        my_ax.add_patch(i)

    plt.axis('equal')
    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.show()


########################################

if __name__ == "__main__":
    source_img = imread("filmrole3_001.jpg")
    target_img = imread("stadium.png")

    source_img = source_img / 255    # scaling the jpg file from 0 to 1 instead of 0 to 255
    target_img = target_img[:,:,:-1] # removing the last component (alpha) of the png file

    blue_heatmap = np.zeros((target_img.shape[0], target_img.shape[1]))
    white_heatmap = np.zeros((target_img.shape[0], target_img.shape[1]))

    H = homography_solve()

    blue_players_video = [[750, 446], [765, 575], [1269, 288], [1422, 684], [1689, 696], [12, 108], [1913, 390]] # last two are fake, just to make the map pretty atm
    white_players_video = [[894, 526], [1454, 696], [1736, 387], [17, 779], [267, 131]] # last two are fake, just to make the map pretty atm
    blue_players_map = []
    white_players_map = []


    for blue_player_video in blue_players_video:
        player_pos = homography_transform(blue_player_video, H)
        blue_players_map.append([int(player_pos[0]), int(player_pos[1])])
        print(blue_players_map[-1])

    for white_player_video in white_players_video:
        player_pos = homography_transform(white_player_video, H)
        white_players_map.append([int(player_pos[1]), int(player_pos[0])])


    plot_heatmap(np.array(blue_players_map)[:,0], np.array(blue_players_map)[:,1], 'blue')
    plot_heatmap(np.array(white_players_map)[:,0], np.array(white_players_map)[:,1], 'red')


#
