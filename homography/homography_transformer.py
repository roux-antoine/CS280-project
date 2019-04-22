import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import seaborn as sns

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


def plot_heatmap(map):

    ax = sns.heatmap(map, cmap="coolwarm")

    height = map.shape[0]
    width = map.shape[1]

    # Prepare Circles
    centreCircle = plt.Circle((width/2, height/2), 100, color="white", fill=False)
    centreSpot = plt.Circle((width/2, height/2), 10, color="white")
    ax.add_patch(centreCircle)
    ax.add_patch(centreSpot)

    plt.axis('equal')
    plt.show()

########################################

if __name__ == "__main__":
    source_img = imread("filmerole3_001.jpg")
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

    for blue_player_video in blue_players_video:
        player_pos = homography_transform(blue_player_video, H)
        blue_players_map.append([int(player_pos[1]), int(player_pos[0])])
        print(blue_players_map[-1])

        for i in range(-10, 10):
            for j in range(-10, 10):
                blue_heatmap[blue_players_map[-1][0] + i][blue_players_map[-1][1] + j] += 1
        # blue_heatmap[blue_players_map[-1][0]-10:blue_players_map[-1][0]+10][blue_players_map[-1][1]-10:blue_players_map[-1][1]+10] += 1


    for white_player_video in white_players_video:
        player_pos = homography_transform(white_player_video, H)
        white_players_map.append([int(player_pos[1]), int(player_pos[0])])

        print(white_players_map[-1])

        for i in range(-10, 10):
            for j in range(-10, 10):
                white_heatmap[white_players_map[-1][0] + i][white_players_map[-1][1] + j] += 1

        # white_heatmap[white_players_map[-1][0]][white_players_map[-1][1]] += 1


    plot_heatmap(blue_heatmap)
    plot_heatmap(white_heatmap)

#
