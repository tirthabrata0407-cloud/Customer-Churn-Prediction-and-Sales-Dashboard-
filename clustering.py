from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA


def scale_features(df, feature_cols):
    scaler = StandardScaler()
    X_s = scaler.fit_transform(df[feature_cols])
    return X_s, scaler


def kmeans_elbow(X_s, k_range=range(2, 9)):
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_s)
        inertias.append(km.inertia_)
    return list(k_range), inertias


def kmeans_fit(X_s, k):
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_s)
    return km


def pca_2d(X_s):
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_s)
    return coords


def dbscan_outliers(df, feature_cols, eps=0.8, min_samples=5, sample_size=5000):
    sample = df.sample(min(sample_size, len(df)), random_state=42)
    X_sample_s, _ = scale_features(sample, feature_cols)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_sample_s)
    n_outliers = int((db.labels_ == -1).sum())
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    return sample, db, n_outliers, n_clusters