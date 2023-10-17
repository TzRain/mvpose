import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

CONNECTIVITY_DICT = {
    'cmu': [(0, 2), (0, 9), (1, 0), (1, 17), (2, 12), (3, 0), (4, 3), (5, 4), (6, 2), (7, 6), (8, 7), (9, 10), (10, 11), (12, 13), (13, 14), (15, 1), (16, 15), (17, 18)],
    'coco': [(0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (5, 11), (6, 12), (11, 12)],
    "mpii": [(0, 1), (1, 2), (2, 6), (5, 4), (4, 3), (3, 6), (6, 7), (7, 8), (8, 9), (8, 12), (8, 13), (10, 11), (11, 12), (13, 14), (14, 15)],
    "human36m": [(0, 1), (1, 2), (2, 6), (5, 4), (4, 3), (3, 6), (6, 7), (7, 8), (8, 16), (9, 16), (8, 12), (11, 12), (10, 11), (8, 13), (13, 14), (14, 15)],
    "kth": [(0, 1), (1, 2), (5, 4), (4, 3), (6, 7), (7, 8), (11, 10), (10, 9), (2, 3), (3, 9), (2, 8), (9, 12), (8, 12), (12, 13)],
}

COLOR_DICT = {
    'coco': [
        (102, 0, 153), (153, 0, 102), (51, 0, 153), (153, 0, 153),  # head
        (51, 153, 0), (0, 153, 0),  # left arm
        (153, 102, 0), (153, 153, 0),  # right arm
        (0, 51, 153), (0, 0, 153),  # left leg
        (0, 153, 102), (0, 153, 153),  # right leg
        (153, 0, 0), (153, 0, 0), (153, 0, 0), (153, 0, 0)  # body
    ],

    'human36m': [
        (0, 153, 102), (0, 153, 153), (0, 153, 153),  # right leg
        (0, 51, 153), (0, 0, 153), (0, 0, 153),  # left leg
        (153, 0, 0), (153, 0, 0),  # body
        (153, 0, 102), (153, 0, 102),  # head
        (153, 153, 0), (153, 153, 0), (153, 102, 0),   # right arm
        (0, 153, 0), (0, 153, 0), (51, 153, 0)   # left arm
    ],

    'kth': [
        (0, 153, 102), (0, 153, 153),  # right leg
        (0, 51, 153), (0, 0, 153),  # left leg
        (153, 102, 0), (153, 153, 0),  # right arm
        (51, 153, 0), (0, 153, 0),  # left arm
        (153, 0, 0), (153, 0, 0), (153, 0, 0), (153, 0, 0), (153, 0, 0), # body
        (102, 0, 153) # head
    ]
}

# size = 1024
# n_cols,n_rows = 1,1
# fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(n_cols * size, n_rows * size),)
# axes = axes.reshape(n_rows, n_cols)
fig=plt.figure()
axes=fig.gca(projection='3d')

def draw_3d_pose(keypoints, ax, keypoints_mask=None, kind='cmu', radius=None, root=None, point_size=2, line_width=2, draw_connections=True):
    
    connectivity = CONNECTIVITY_DICT[kind]

    if keypoints_mask is None:
        keypoints_mask = [True] * len(keypoints)

    if draw_connections:
        # Make connection matrix
        for i, joint in enumerate(connectivity):
            if keypoints_mask[joint[0]] and  keypoints_mask[joint[1]]:
                xs, ys, zs = [np.array([keypoints[joint[0], j], keypoints[joint[1], j]]) for j in range(3)]

                if kind in COLOR_DICT:
                    color = COLOR_DICT[kind][i]
                else:
                    color = (0, 0, 255)

                color = np.array(color) / 255

                ax.plot(xs, ys, zs, lw=line_width, c=color)

        if kind == 'coco':
            mid_collarbone = (keypoints[5, :] + keypoints[6, :]) / 2
            nose = keypoints[0, :]

            xs, ys, zs = [np.array([nose[j], mid_collarbone[j]]) for j in range(3)]

            if kind in COLOR_DICT:
                color = (153, 0, 51)
            else:
                color = (0, 0, 255)

            color = np.array(color) / 255

            ax.plot(xs, ys, zs, lw=line_width, c=color)


    ax.scatter(keypoints[keypoints_mask][:, 0], keypoints[keypoints_mask][:, 1], keypoints[keypoints_mask][:, 2],
               s=point_size, c=np.array([230, 145, 56])/255, edgecolors='black')  # np.array([230, 145, 56])/255

    if radius is not None:
        if root is None:
            root = np.mean(keypoints, axis=0)
        xroot, yroot, zroot = root
        ax.set_xlim([-radius + xroot, radius + xroot])
        ax.set_ylim([-radius + yroot, radius + yroot])
        ax.set_zlim([-radius + zroot, radius + zroot])

    ax.set_aspect('equal')

keypoints = np.load('poses3d_cmu.npy')

# print(keypoints)

for keypoint in keypoints:
    draw_3d_pose(keypoint.T,ax=axes,kind='coco')

plt.savefig('./pose.png')
plt.title('Extrinsic Parameters')
plt.show()