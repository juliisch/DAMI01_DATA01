"""
Digital Business University of Applied Sciences
Data Science und Management (M. Sc.)
DAMI01 / DATA01 Data Analytics
Prof. Dr. Daniel Ambach
Julia Schmid (200022)

In dieser Datei sind die Funktionen definiert.
"""

# ---------- Imports ----------
import numpy as np
from kneed import KneeLocator
from sklearn.cluster import AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn_extra.cluster import KMedoids
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import cdist_dtw


# ---------- ZUSATZFUNKTIONEN ----------
def get_key_value(value, dictionary):
    """
    Funktion:         Bestimmt den Key-Wert vom Dictionary
    Input:            value (Wert-Element)
                      dictionary (Dictionary)
    Output:           key_value (Key-Wert)
    Funktionsweise:   Basierend auf dem übergebenen Dictionaries werden
                      die Wert-Listen im Dictionary durchsucht und den
                      zugehörigen Key zurückgegeben
    """
    key_value = None
    for key, sublist in dictionary.items():
        if value in sublist:
            key_value = key
    return key_value


# ---------- MODELL-VORBEREITUNG ----------
def get_dtw(x_ts):
    """
    Funktion:         Berechnet DTW-Distanzmatrizen
    Input:            x_ts (skalierte und transformierte Zeitreihen)
    Output:           d_dict (Dictionary mit drei DTW-Distanzmatrizen)
    Funktionsweise:   Berechnet für die übergebenen Zeitreihen paarweise
                      DTW-Distanzen in drei Varianten: ohne Einschränkung,
                      mit Sakoe-Chiba-Band und mit Itakura-Parallelogramm.
    """
    # Standard DTW
    print(" [Info] Berechne DTW Standard")
    d_stand = cdist_dtw(x_ts, n_jobs=-1)

    # DTW mit Sakoe-Chiba Band
    print(" [Info] Berechne DTW mit Sakoe-Chiba Band")
    d_sakoe = cdist_dtw(x_ts, sakoe_chiba_radius=10, n_jobs=-1)

    # DTW mit Itakura Parallelogramm
    print(" [Info] Berechne DTW mit Itakura Parallelogramm")
    d_itakura = cdist_dtw(x_ts, itakura_max_slope=2.0, n_jobs=-1)

    # Dictionary der drei DTW-Matritzen
    d_dict = {
        "standard": d_stand,
        "sakoe": d_sakoe,
        "itakura": d_itakura,
    }

    return d_dict


def dtw_transformation(d):
    """
    Funktion:         Transformiert eine Distanzmatrix in eine
                      Ähnlichkeitsmatrix
    Input:            d (Distanzmatrix)
    Output:           a (Ähnlichkeitsmatrix)
    Funktionsweise:   Für die Tranformation der Distanzmatrix in eine
                      Ähnlichkeitsmatrix wird der Gauß-Kernel (RBF)
                      [s_ij = exp(-d_ij**2/2*sigma**2),
                      wobei sigma = Median(d_ij) für d_ij > 0 ]
                      verwendet.
    """
    sigma = np.median(d[d > 0])  # sigma = Median(d_ij) für d_ij > 0
    a = np.exp(-(d ** 2) / (2 * sigma ** 2))
    return a


# ---------- MODELLE ----------
def model_baseline(k):
    """
    Funktion:         Erstellung eines Baseline-k-Means-Modells
    Input:            k (Anzahl der Cluster)
    Output:           model_baseline (Baseline-k-Means-Modell Modell)
    Funktionsweise:   Erzeugt ein Baseline-k-Means-Modell mit der
                      angebebnen Clusteranzahl
    """
    model = TimeSeriesKMeans(
        n_clusters=k,
        metric="euclidean",
        random_state=123,
        n_init=10,
    )
    return model


def model_agglomerative(k, linkage):
    """
    Funktion:         Erstellung eines Agglomerative-Clustering-Modells
    Input:            k (Anzahl der Cluster)
                      linkage (Linkage-Methode)
    Output:           model_agglomerative (Agglomerative-Clustering-Modell)
    Funktionsweise:   Erzeugt ein Agglomerative-Clustering-Modell mit
                      der angegebenen Clusteranzahl und Linkage-Methode.
    """
    model = AgglomerativeClustering(
        n_clusters=k,
        metric="precomputed",
        linkage=linkage,
    )
    return model


def model_spectral(k, assign_label):
    """
    Funktion:         Erstellung eines Spectral-Clustering-Modells
    Input:            k (Anzahl der Cluster)
                      assign_label (Zuordnungsstrategie)
    Output:           model_spectral (Spectral-Clustering-Modell)
    Funktionsweise:   Erzeugt ein Spectral-Clustering-Modell mit
                      der angegebenen Clusteranzahl und
                      Zuordnungsstrategie
    """
    model = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        assign_labels=assign_label,
        random_state=123,
    )
    return model


def model_kmedoids(k, init):
    """
    Funktion:         Erstellung eines K-Medoids-Modells
    Input:            k (Anzahl der Cluster)
                      init (Initialisierungsmethode)
    Output:           model_kmedoids (K-Medoids-Modell)
    Funktionsweise:   Erzeugt ein K-Medoids-Modell mit
                      der angegebenen Clusteranzahl und
                      Initialisierungsmethode
    """
    model = KMedoids(
        n_clusters=k,
        metric="precomputed",
        init=init,
        random_state=123,
    )
    return model


def model_dbscan(eps_stand, min_samples_var):
    """
    Funktion:         Erstellung eines DBSCAN Modells
    Input:            eps_stand (Epsilon)
                      min_samples_var (Mindestanzahl)
    Output:           model_dbscan (DBSCAN Modell)
    Funktionsweise:   Erzeugt ein DBSCAN-Modell mit
                      der angegebenen Epsilon und Mindestanzahl
    """
    model = DBSCAN(
        eps=eps_stand,
        min_samples=min_samples_var,
        metric="precomputed",
    )
    return model


def model_fcm(d, n_clusters, m, max_iteration=100, konvergenz=1e-5):
    """
    Funktion:         Erstellt ein Fuzzy C-Medoids auf Basis einer
                      Distanzmatrix
    Input:            d (Distanzmatrix)
                      n_clusters (Anzahl der Cluster)
                      m (Fuzziness-Exponent)
                      max_iteration (maximale Iterationsanzahl)
                      konvergenz (Konvergenz-Schwellenwert)
    Output:           u (Zugehörigkeitsmatrix)
    Funktionsweise:   Zunächst wird eine zufällige Zugehörigkeitsmatrix
                      initialisiert und iterativ optimiert.
                      Dazu werden iterativ die Cluster-Zentretn
                      aktualisiert und die Zugehörigkeiten aktualisiert.
                      Nach dem erreichen des Konvergenz-Schwellenwert
                      oder der maximale Iterationsanzahl wird der Algo
                      abgebrochen.
    """
    n = d.shape[0]
    rng = np.random.default_rng(123)

    # Initiale Clusterzentren zufällig wählen
    centers_idx = rng.choice(n, n_clusters, replace=False)

    # Initiale Membership-Matrix aus den Zentren berechnen
    u = np.zeros((n, n_clusters))
    for i in range(n):
        d_i = d[i, centers_idx]
        d_i = np.maximum(d_i, 1e-10)
        for c in range(n_clusters):
            u[i, c] = 1.0 / np.sum(
                (d_i[c] / d_i) ** (2 / (m - 1))
            )

    for iteration in range(max_iteration):
        u_old = u.copy()

        # Cluster-Zentren bestimmen
        centers_idx = []
        for c in range(n_clusters):
            weights = u[:, c] ** m
            costs = d.dot(weights)
            centers_idx.append(np.argmin(costs))
        centers_idx = np.array(centers_idx)

        # Cluster-Zugehörigkeiten bestimmen
        for i in range(n):
            d_i = d[i, centers_idx]
            d_i = np.maximum(d_i, 1e-10)
            for c in range(n_clusters):
                u[i, c] = 1.0 / np.sum(
                    (d_i[c] / d_i) ** (2 / (m - 1))
                )

        # Abbruchbedingung prüfen
        differenz = np.linalg.norm(u - u_old)
        if differenz < konvergenz:
            break

    return u
# Quelle:
# Krishnapuram, R., Joshi, A., Nasraoui, O., & Yi, L. (2001).
# Low-complexity fuzzy relational clustering algorithms for Web mining.
# IEEE transactions on Fuzzy Systems, 9(4), 595–607.

# ---------- MODELL-HILFSFUNKTIONEN ----------

def dunn_index_dtw(d, labels):
    """
    Funktion:         Bestimmung des Dunn-Index für DTW Distanz
    Input:            d (Distanzmatrix)
                      labels (Cluster-Zuordnung)
    Output:           dunn_index (Dunn-Index)
    Funktionsweise:   Bestimmung des Dunn-Index gemäß
            DI = min(delta(C_i, C_j)) / max(delta(C_l))
    """
    clusters = [np.where(labels == l)[0] for l in np.unique(labels)]
    n_clusters = len(clusters)

    # Zähler: min(Delta(C_i, C_j))
    # (kleinster Abstand zwischen zwei Clustern)
    delta_cluster_distance = np.inf
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            dist_ci_cj = d[np.ix_(clusters[i], clusters[j])]
            delta_cluster_distance = min(
                delta_cluster_distance, np.min(dist_ci_cj)
            )

    # Nenner: max(Delta(C_l))
    # (Größter Durchmesser von allen Clustern)
    delta_cluster_diameter = 0
    for i in range(n_clusters):
        if len(clusters[i]) <= 1:
            continue
        dist_int = d[np.ix_(clusters[i], clusters[i])]
        delta_cluster_diameter = max(
            delta_cluster_diameter, np.max(dist_int)
        )

    # Dunn-Index
    dunn_index = delta_cluster_distance / delta_cluster_diameter

    return dunn_index
# Quelle: Ikotun, A. M., Habyarimana, F., & Ezugwu, A. E. (2025).
# Cluster validity indices for automatic: A comprehensive review.
# Heliyon, 11(2)


def davies_bouldin_dtw(d, labels):
    """
    Funktion:         Berechnet den Davies-Bouldin-Index
    Input:            d (Distanzmatrix)
                      labels (Cluster-Zuordnung)
    Output:           davies_bouldin_index (Davies-Bouldin-Index)
    Funktionsweise:   Bestimmung des Davies-Bouldin-Index gemäß
                      DBI = (1/k) * Σ max(R_ij) mit
                      R_ij = (S_i + S_j) / M_ij
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    s = np.zeros(n_clusters)
    medoids = np.zeros(n_clusters, dtype=int)

    for i, k in enumerate(unique_labels):
        members = np.where(labels == k)[0]

        if len(members) <= 1:
            s[i] = 0
            medoids[i] = members[0]
            continue

        sub_d = d[np.ix_(members, members)]
        medoid_local = np.argmin(sub_d.mean(axis=1))
        medoids[i] = members[medoid_local]
        s[i] = sub_d[medoid_local].mean()

    r = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i != j:
                dist = d[medoids[i], medoids[j]]
                r[i, j] = (s[i] + s[j]) / dist if dist > 0 else 0

    davies_bouldin_index = np.mean([
        np.max(r[i, np.arange(n_clusters) != i])
        for i in range(n_clusters)
    ])

    return davies_bouldin_index


def evaluate_cluster(d, labels):
    """
    Funktion:         Evaluation der Clusterergebnisse
    Input:            d (Distanzmatrix),
                      labels (Cluster-Zuordnung)
    Output:           sil_score (Silhouette-Score)
                      dbi_score (Davies-Bouldin-Index)
                      dunn_index_score (Dunn-Index)
    Funktionsweise:   Abhängig von den übergebendenen Labels und
                      Distanzmatrix wird der Silhoutten-Score,
                      Davies-Bouldin-Index und der Dunn-Index
                      bestimmt und zurückgegeben.
    """
    if len(set(labels)) < 2:
        return np.nan, np.nan, np.nan

    sil_score = round(
        silhouette_score(d, labels, metric="precomputed"), 2
    )
    dbi_score = round(davies_bouldin_dtw(d, labels), 2)
    dunn_index_score = round(dunn_index_dtw(d, labels), 2)

    return sil_score, dbi_score, dunn_index_score


def find_optimal_eps(d, k):
    """
    Funktion:         Bestimmt den optimalen Epsilon-Wert
    Input:            d (Distanzmatrix)
                      k (Anzahl der Nachbarn)
    Output:           eps (Epsilon)
    Funktionsweise:   Für jeden Punkt wird die Distanz zum
                      k-nöchsten Nachbarn mittels NearestNeighbors,
                      basierend auf der Distanzmatrix bestimmt,
                      Anschließend werden die k-Distanzen aufsteigend
                      sortiert. Der Kniepunkt der Kurve wird mit dem
                      KneeLocators ermittelt und als optimaler
                      Epsilon-Wert zurückgegeben.
    """
    model_nn = NearestNeighbors(n_neighbors=k, metric="precomputed")
    distances, _ = model_nn.fit(d).kneighbors(d)
    k_distances = np.sort(distances[:, k - 1])

    knee = KneeLocator(
        range(len(k_distances)),
        k_distances,
        curve="convex",
        direction="increasing",
    )

    eps = k_distances[knee.knee]
    return eps


def find_opt_params(d, model_name, a=None, x_ts=None):
    """
    Funktion:         Bestimmung der optimalen Parameter
    Input:            d (Distanzmatrix)
                      model_name (Names des Modells)
                      a (Affinitätsmatrix)
                      x_ts (Zeitreihen im tslearn-Format)
    Output:           best_params (optimale Parameter)
    Funktionsweise:   Es werden für das übergebende Modell
                      verschiedene Clusteranzahlen k sowie
                      weitere Modell-spezifische Parameter getestet.
                      Dazu werden die verscheidenen
                      Parameter-Kombinationen auf das Modell
                      angewendet und dafür den Silhouette-Score
                      bestimmt. Die Parameter mit den höchsten
                      Score, werden zurückgegeben.
    """
    list_k = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    best_score = -1
    best_params = {}

    if model_name == "kmedoids":
        for k in list_k:
            for init in ["k-medoids++", "heuristic", "random"]:
                labels = model_kmedoids(k, init).fit_predict(d)
                score = silhouette_score(
                    d, labels, metric="precomputed"
                )
                if score > best_score:
                    best_score = score
                    best_params = {"k": k, "init": init}

    if model_name == "agglomerative":
        for k in list_k:
            for linkage in ["average", "complete", "single"]:
                labels = model_agglomerative(
                    k, linkage
                ).fit_predict(d)
                score = silhouette_score(
                    d, labels, metric="precomputed"
                )
                if score > best_score:
                    best_score = score
                    best_params = {"k": k, "linkage": linkage}

    if model_name == "spectral":
        for k in list_k:
            for assign_label in ["kmeans", "discretize"]:
                labels = model_spectral(
                    k, assign_label
                ).fit_predict(a)
                score = silhouette_score(
                    d, labels, metric="precomputed"
                )
                if score > best_score:
                    best_score = score
                    best_params = {
                        "k": k,
                        "assign_label": assign_label,
                    }

    if model_name == "fcm":
        for k in list_k:
            for m in [1.5, 2.0, 2.5, 3.0]:
                u = model_fcm(d, n_clusters=k, m=m)
                labels = np.argmax(u, axis=1)
                if len(set(labels)) < 2:
                    continue
                score = silhouette_score(
                    d, labels, metric="precomputed"
                )
                if score > best_score:
                    best_score = score
                    best_params = {"k": k, "m": m}

    return best_params


def run_model(
    clustering_model_name,
    d_dict,
    df_evaluate,
    n_run,
    x_scaled=None,
    x_ts=None,
):
    """
    Funktion:       Wendet eine Clusteringmethode auf alle
                    DTW-Varianten an und evaluiert die Ergebnisse
    Input:          clustering_model_name (Name der Clusteringmethode)
                    d_dict (Dictionary mit DTW-Distanzmatrizen)
                    df_evaluate (Evaluationsergebnisse-DataFrame)
                    n_run (Durchgangsnummer)
                    x_scaled (skalierte Zeitreihen)
    Output:         dict_labels (Dictionary mit Cluster-Zuordnungen
                    pro DTW-Variante)
    Funktionsweise: Jedes Modell wird über alle DTW-Varianten
                    Iteriert. Dabei werden die optimalen Parameter
                    bestimmt, die Modelle angewendet, die
                    Evaluationskennzahlen bestimmt und im
                    Evaluationsergebnisse-DataFrame gespeichert sowie
                    die Cluster-Zuordnungen gespeichert.
    """
    print(
        f" [Info] Starte Clusteringmethode {clustering_model_name}"
    )
    dict_labels = {}
    d_dict = {key: np.asarray(val) for key, val in d_dict.items()}

    if clustering_model_name == "baseline":
        labels = model_baseline(2).fit_predict(x_ts)
        for d_kind, d in d_dict.items():
            df_evaluate.loc[len(df_evaluate)] = [
                n_run,
                clustering_model_name,
                d_kind,
                2,
                *evaluate_cluster(d, labels),
            ]
            dict_labels[d_kind] = labels

    if clustering_model_name == "kmedoids":
        for d_kind, d in d_dict.items():
            best_params = find_opt_params(d, clustering_model_name)
            labels = model_kmedoids(
                best_params["k"], init=best_params["init"]
            ).fit_predict(d)
            df_evaluate.loc[len(df_evaluate)] = [
                n_run,
                clustering_model_name,
                d_kind,
                best_params["k"],
                *evaluate_cluster(d, labels),
            ]
            dict_labels[d_kind] = labels

    if clustering_model_name == "agglomerative":
        for d_kind, d in d_dict.items():
            best_params = find_opt_params(d, clustering_model_name)
            labels = model_agglomerative(
                best_params["k"], best_params["linkage"]
            ).fit_predict(d)
            df_evaluate.loc[len(df_evaluate)] = [
                n_run,
                clustering_model_name,
                d_kind,
                best_params["k"],
                *evaluate_cluster(d, labels),
            ]
            dict_labels[d_kind] = labels

    if clustering_model_name == "spectral":
        for d_kind, d in d_dict.items():
            a = dtw_transformation(d)
            best_params = find_opt_params(
                d, clustering_model_name, a
            )
            labels = model_spectral(
                best_params["k"], best_params["assign_label"]
            ).fit_predict(a)
            df_evaluate.loc[len(df_evaluate)] = [
                n_run,
                clustering_model_name,
                d_kind,
                best_params["k"],
                *evaluate_cluster(d, labels),
            ]
            dict_labels[d_kind] = labels

    if clustering_model_name == "fcm":
        for d_kind, d in d_dict.items():
            best_params = find_opt_params(d, clustering_model_name)
            u = model_fcm(
                d, n_clusters=best_params["k"], m=best_params["m"]
            )
            labels = np.argmax(u, axis=1)
            n_clusters = len(set(labels))
            df_evaluate.loc[len(df_evaluate)] = [
                n_run,
                clustering_model_name,
                d_kind,
                best_params["k"],
                *evaluate_cluster(d, labels),
            ]
            dict_labels[d_kind] = labels

    if clustering_model_name == "dbscan":
        for d_kind, d in d_dict.items():
            # Optimale Parameter bestimmen
            min_samples_var = int(np.log(len(x_scaled)))
            eps_stand = round(find_optimal_eps(d, min_samples_var), 0)
            labels = model_dbscan(
                eps_stand, min_samples_var
            ).fit_predict(d)
            n_clusters = len(set(labels))
            df_evaluate.loc[len(df_evaluate)] = [
                n_run,
                clustering_model_name,
                d_kind,
                n_clusters,
                *evaluate_cluster(d, labels),
            ]
            dict_labels[d_kind] = labels

    return dict_labels