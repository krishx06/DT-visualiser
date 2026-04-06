import numpy as np
from sklearn.datasets import make_classification, make_moons, make_circles

# ---------------- DATA ----------------
def generate_data(dataset_type="moons", noise=0.2):
    if dataset_type == "simple":
        X, y = make_classification(
            n_features=2,
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=42
        )
    elif dataset_type == "moons":
        X, y = make_moons(noise=noise, random_state=42)
    elif dataset_type == "circles":
        X, y = make_circles(noise=noise, factor=0.5, random_state=42)
    return X, y


# ---------------- IMPURITY ----------------
def gini(y):
    classes = np.unique(y)
    impurity = 1
    for c in classes:
        p = np.sum(y == c) / len(y)
        impurity -= p ** 2
    return impurity


def entropy(y):
    classes = np.unique(y)
    ent = 0
    for c in classes:
        p = np.sum(y == c) / len(y)
        ent -= p * np.log2(p + 1e-9)
    return ent