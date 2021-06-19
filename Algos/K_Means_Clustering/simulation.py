import numpy as np
import time

def kMean(K: int,
          data: np.ndarray,
          max_epochs=10,
          delay=100,
          seed=-1):
    """
    :param int K: (Desired # of clusters)
    :param np.ndarray data: (
        n x d matrix,
        where n: number of coordinated and d: number of dimensions
    )
    :param int max_epochs: (default: 10)
    :param int delay: (delay(ms) in yield default: 100ms)
    :param int seed: (for reproducibility default: -1)
    :return labels: (Cluster Label for each data point)
    """

    if seed != -1:
        np.random.seed(seed)
    else:
        np.random.seed()

    n, dim = data.shape  # n: number of data points, dim: dimension
    centroids = []  # centroids: centroid of K clusters
    low, high = np.min(data), np.max(data)

    # First let's randomly assign centroids for all K clusters
    for d in range(dim):
        z_ds = np.random.uniform(low=low, high=high, size=K)
        centroids.append(z_ds)
    centroids = np.array(centroids).T

    # `labels` will be the labels we assign to each data point which will represent there cluster
    labels = np.zeros(shape=n, dtype=int)
    OLD_COST = 0

    for _ in range(max_epochs):

        # cardinalities will be our |C_j|'s OR say number of data points in Cluster 'j'
        cardinalities = np.zeros(shape=K, dtype=int)

        # Assign each data point data[i] to the closest z[j]
        NEW_COST = 0
        for i in range(n):

            # Calculating L2 (euclidean) distance of the ith data point to all centroids
            L2 = (centroids - data[i]) ** 2
            L2 = L2.sum(axis=1)

            # Choose the closest centroid
            idx = L2.argmin()
            labels[i] = idx

            # Update the overall cost for ith data
            NEW_COST += L2[idx]

            # Increment Number of data points for the chosen cluster
            cardinalities[labels[i]] += 1

            if i % (n // 30) == 0:
                time.sleep(delay/1000)
                yield labels

        if NEW_COST == OLD_COST:
            break
        else:
            OLD_COST = NEW_COST

        centroids = np.zeros(shape=(K, dim))

        # Now we have some data points for some clusters
        # So let's find the best representatives of z[1],...,z[K]
        # We will get the centroid for all cluster, by there respective data points and use then as new centroids.
        for i in range(n):
            group = labels[i]
            centroids[group] = centroids[group] + data[i]
        centroids = np.array(
            [centroid / cardinalities[j] if cardinalities[j] != 0 else centroid
                for j, centroid in enumerate(centroids)]
        )

        # Now let's give friends to lonely centroids
        # Here we cover those z[j] 's who haven't get any data points
        for j in range(K):
            if cardinalities[j] == 0:

                # `distances` is cumulative distance of each data point from all centroids
                distances = np.array([np.nansum(np.sum((centroids - data[i]) ** 2, axis=1)) for i in range(n)])

                # Now give high weight to most lonely data point
                loneliness = distances / np.max(distances)

                # Now take the weighted average of all data points based on there loneliness
                centroids[j] = ((data * np.array([loneliness]).T).sum(axis=0)) / loneliness.sum()

    return labels
