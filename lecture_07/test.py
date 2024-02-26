import numpy as np
from PIL import Image as im
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from sklearn.neighbors import NearestNeighbors

TEMPFILE = 'temp.png'

class DBC():

    def __init__(self, dataset, min_pts, epsilon):
        self.dataset = dataset
        self.min_pts = min_pts
        self.epsilon = epsilon
        self.labels = np.zeros(len(dataset), dtype=int) - 1
        self.snaps = []
        self.neigh = NearestNeighbors(radius=self.epsilon).fit(dataset)

    def snapshot(self, idx):
        # Use a consistent color sequence for clusters, noise points remain black
        colors = np.array(['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'black'])
        # Create a circle around the current point
        fig, ax = plt.subplots()
        for i in range(len(self.dataset)):
            if self.labels[i] == -1:
                ax.plot(self.dataset[i,0], self.dataset[i,1], 'ok', markersize=6)
            else:
                ax.plot(self.dataset[i,0], self.dataset[i,1], 'o', markerfacecolor=colors[self.labels[i] % len(colors)], markersize=6)
        cir = plt.Circle((self.dataset[idx,0], self.dataset[idx,1]), self.epsilon, color='black', fill=False)
        ax.add_patch(cir)
        ax.set_xlim(-3, 4)
        ax.set_ylim(-2, 4)
        ax.set_aspect('equal')
        fig.savefig(TEMPFILE)
        plt.close(fig)
        self.snaps.append(im.open(TEMPFILE).convert('RGB'))

    def _region_query(self, point_idx):
        return self.neigh.radius_neighbors([self.dataset[point_idx]], return_distance=False)[0]

    def _expand_cluster(self, point_idx, neighbors, cluster_id):
        self.labels[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            p = neighbors[i]
            if self.labels[p] == -1:
                self.snapshot(p)
                self.labels[p] = cluster_id
                p_neighbors = self._region_query(p)
                if len(p_neighbors) >= self.min_pts:
                    neighbors = np.append(neighbors, p_neighbors)
            elif self.labels[p] == 0:
                self.labels[p] = cluster_id
            i += 1

    def dbscan(self):
        cluster_id = 0
        for point_idx in range(self.dataset.shape[0]):
            if self.labels[point_idx] != -1:
                continue
            neighbors = self._region_query(point_idx)
            if len(neighbors) < self.min_pts:
                self.labels[point_idx] = -1  # Noise
            else:
                cluster_id += 1
                self._expand_cluster(point_idx, neighbors, cluster_id)
        return self.labels

# Create the 'face' dataset
centers = [[-1, 2], [1, 2]]
eyes, _ = datasets.make_blobs(n_samples=150, centers=centers, cluster_std=0.3)

mouth_x = np.linspace(-2, 2, 100)
mouth_y = 0.2 * mouth_x**2 + 0.02 * np.random.randn(mouth_x.shape[0])
mouth = np.vstack((mouth_x, mouth_y)).T

face = np.append(eyes, mouth, axis=0)

# Perform DBSCAN clustering
dbc = DBC(face, 3, 0.3)
clustering = dbc.dbscan()

# Save the snapshots as a GIF
gif_filename = 'dbscan.gif'
dbc.snaps[0].save(
    gif_filename,
    optimize=False,
    save_all=True,
    append_images=dbc.snaps[1:],
    duration=25,
    loop=0
)

