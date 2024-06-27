import numpy as np

def minkowski_dist(s1, s2, n=2):
    s1 = np.array(s1)
    s2 = np.array(s2)
    distances_abs = np.abs(s2-s1)
    parcels = distances_abs**n
    the_sum = parcels.sum()
    dist = the_sum ** (1/n)
    return dist
# def minkowski_dist


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))
# def euclidean_distance

class KNN:
    def __init__(
        self,
        p_k=3,
        p_dist_metric=minkowski_dist
    ):
        self.mK = p_k
        self.mDistMetric = p_dist_metric
    # def __init__

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    # def fit

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    # def

    # "private"
    def _predict(self, p_sample):
        # Compute distances between x and all examples in the training set
        distances = [self.mDistMetric(p_sample, x_train) for x_train in self.X_train]

        # argsort returns the indices that would sort an array.
        k_indices_for_the_shortest_distances = np.argsort(distances)[:self.mK] # sort, then k-first only

        # Extract the labels of the nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices_for_the_shortest_distances]

        # form a set, apply
        most_common = max(
            set(k_nearest_labels), # no repetitions, but the point is to consider each label only once, not more
            key=k_nearest_labels.count # this "count" is a function that will work on k_nearest_labels (not an int)
        )
        return most_common # some label
    # def _predict
# KNN

# usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification # for artificial datasets
    from sklearn.model_selection import train_test_split

    # generating an artificial dataset and its labels
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=9,
        n_redundant=1,
        n_repeated=0,
        random_state=42
    )

    X_train, X_test, y_train, y_test =\
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=123
        )

    my_knn = KNN(
        p_k=1
    )
    my_knn.fit(X_train, y_train)
    predictions = my_knn.predict(X_test)

    # accuracy
    print(f"accuracy: {np.mean(predictions == y_test)}")
# if
