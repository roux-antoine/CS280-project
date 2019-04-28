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

def plot_heatmap(map):

    # height = map.shape[0]
    # width = map.shape[1]
    #
    # x_coord = [1269, 40, 900]
    # y_coord = [288, 40, 900]
    #
    #
    # my_ax = sns.kdeplot(x_coord, y_coord, shade = "True", color = "green", n_levels = 30)
    #
    # pitch = plt.Rectangle([0, 0], width = width, height = height, fill = False)
    # leftPenalty = plt.Rectangle([0, 0.28*height], width = 0.12*width, height = 0.45*height, fill = False)
    # rightPenalty = plt.Rectangle([0.88*width, 0.28*height], width = 0.12*width, height = 35.3/80*height, fill = False)
    # midline = matplotlib.patches.ConnectionPatch([0.5*width, 0], [0.5*width, height], "data", "data")
    #
    # #Left, Right 6-yard Box
    # leftSixYard = plt.Rectangle([0, 0.4*height], width = 0.04*width, height = 0.2*height, fill = False)
    # rightSixYard = plt.Rectangle([0.96*width, 0.4*height], width = 0.04*width, height = 0.2*height, fill = False)
    #
    # # #Prepare Circles
    # centreCircle = plt.Circle((0.5*width, 0.5*height), 0.09*width,color="black", fill = False)
    # centreSpot = plt.Circle((0.5*width, 0.5*height), 0.005*width,color="black")
    # #Penalty spots and Arcs around penalty boxes
    # leftPenSpot = plt.Circle((0.08*width, 0.5*height), 0.005*width, color="black")
    # rightPenSpot = plt.Circle((0.92*width, 0.5*height), 0.005*width, color="black")
    #
    # element = [pitch, leftPenalty, rightPenalty, midline, leftSixYard, rightSixYard, centreCircle, centreSpot, rightPenSpot, leftPenSpot]
    #
    # for i in element:
    #     my_ax.add_patch(i)
    #
    # plt.axis('equal')
    # plt.show()

    height = map.shape[0]
    width = map.shape[1]



    ax = sns.heatmap(map, cmap="coolwarm")

    ## begin of image border lines ##
    lineCenter1 = matplotlib.patches.ConnectionPatch([width/4, 0], [width/3, height], "data", "data")
    lineCenter2 = matplotlib.patches.ConnectionPatch([3*width/4, 0], [2*width/3, height], "data", "data")
    lineLeft = matplotlib.patches.ConnectionPatch([width/2, 0], [width/3, height], "data", "data")
    lineRight = matplotlib.patches.ConnectionPatch([width/2, 0], [2*width/3, height], "data", "data")
    ax.plot([width/4,   width/3],   [0, height], '--', c='k')  # line center 1
    ax.plot([3*width/4, 2*width/3], [0, height], '--', c='k')  # line center 2
    ax.plot([width/2,   width/3],   [0, height], '--', c='k')  # line left
    ax.plot([width/2,   2*width/3], [0, height], '--', c='k')  # line right

    ## end of image border lines ##

    pitch = plt.Rectangle([0, 0], width = width, height = height, fill = False)
    leftPenalty = plt.Rectangle([0, 0.28*height], width = 0.12*width, height = 0.45*height, fill = False)
    rightPenalty = plt.Rectangle([0.88*width, 0.28*height], width = 0.12*width, height = 35.3/80*height, fill = False)
    midline = matplotlib.patches.ConnectionPatch([0.5*width, 0], [0.5*width, height], "data", "data")

    #Left, Right 6-yard Box
    leftSixYard = plt.Rectangle([0, 0.4*height], width = 0.04*width, height = 0.2*height, fill = False)
    rightSixYard = plt.Rectangle([0.96*width, 0.4*height], width = 0.04*width, height = 0.2*height, fill = False)

    # #Prepare Circles
    centreCircle = plt.Circle((0.5*width, 0.5*height), 0.09*width,color="black", fill = False)
    centreSpot = plt.Circle((0.5*width, 0.5*height), 0.005*width,color="black")
    #Penalty spots and Arcs around penalty boxes
    leftPenSpot = plt.Circle((0.08*width, 0.5*height), 0.005*width, color="black")
    rightPenSpot = plt.Circle((0.92*width, 0.5*height), 0.005*width, color="black")

    element = [pitch, leftPenalty, rightPenalty, midline, leftSixYard, rightSixYard, centreCircle, centreSpot, rightPenSpot, leftPenSpot]

    # bbox = planar.BoundingBox([(0,0), (0, width), (0, height), (width,height)])
    # ax.viewLim = bbox

    for i in element:
        ax.add_patch(i)

    plt.axis('equal')
    plt.show()

########################################

if __name__ == "__main__":
    source_img = imread("filmrole3_001.jpg")
    target_img = imread("stadium.png")

    source_img = source_img / 255 #scaling the jpg file from 0 to 1 instead of 0 to 255
    target_img = target_img[:,:,:-1] #removing the last component (alpha) of the png file

    blue_heatmap = np.zeros((target_img.shape[0], target_img.shape[1]))
    white_heatmap = np.zeros((target_img.shape[0], target_img.shape[1]))

    H = homography_solve()

    blue_players_video = [[1269, 288]]
    white_players_video = [[894, 526]]
    blue_players_map = []
    white_players_map = []

    half_patch_size = 15
    patch = create_gaussian_patch(2*half_patch_size+1, int(half_patch_size/3))

    for blue_player_video in blue_players_video:
        player_pos = homography_transform(blue_player_video, H)
        blue_players_map.append([int(player_pos[1]), int(player_pos[0])])
        print(blue_players_map[-1])

        for i in range(-half_patch_size, half_patch_size):
            for j in range(-half_patch_size, half_patch_size):
                blue_heatmap[blue_players_map[-1][0] + i][blue_players_map[-1][1] + j] += patch[i+half_patch_size, j+half_patch_size]

    for white_player_video in white_players_video:
        player_pos = homography_transform(white_player_video, H)
        white_players_map.append([int(player_pos[1]), int(player_pos[0])])

        print(white_players_map[-1])

        for i in range(-half_patch_size, half_patch_size):
            for j in range(-half_patch_size, half_patch_size):
                white_heatmap[white_players_map[-1][0] + i][white_players_map[-1][1] + j] += patch[i+half_patch_size, j+half_patch_size]


    plot_heatmap(blue_heatmap)
    plot_heatmap(white_heatmap)

#
