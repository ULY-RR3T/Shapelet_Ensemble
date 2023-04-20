from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from k_means_constrained import KMeansConstrained

def make_cluster(k, d, method):
    num_models = d.shape[0]
    if method.lower() == 'agglomerative':
        clustering = AgglomerativeClustering(n_clusters=k,affinity='precomputed',linkage='average').fit(d)
    elif method.lower() == 'dbscan':
        clustering = DBSCAN(eps = k,metric='precomputed',min_samples=5).fit(d)
    elif method.lower() == 'constrained':
        clustering = KMeansConstrained(n_clusters=k,size_min=3).fit(d)
    return clustering.labels_