"""
Digital Business University of Applied Sciences
Data Science und Management (M. Sc.)
DAMI01 / DATA01 Data Analytics
Prof. Dr. Daniel Ambach
Julia Schmid (200022)


In dieser Datei sind die Funktionen definiert. 
"""

# Imports
import numpy as np
from kneed import KneeLocator
from sklearn_extra.cluster import KMedoids
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from tslearn.metrics import cdist_dtw
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
)
from parameter import *

# Funktion:         Bestimmt den Key-Wert vom Dictionary
# Input:            value (Wert-Element)
#                   dictonary (Dictionary)
# Output:           key_value (Key-Wert)
# Funktionsweise:   Basierend auf dem übergebenen Dictionaries werden die Wert-Listen im Dictionary durchsucht und den zugehörigen Key zurückgegeben  
def get_key_value(value, dictonary):
    key_value = None
    for key, sublist in dictonary.items():
        if value in sublist:
            key_value = key
    return key_value  

# ---------- MODELL-VORBEREITUNG ----------
# Funktion:         Berechnet DTW-Distanzmatrizen
# Input:            X_ts (skalierte und transformierte Zeitreihen)
# Output:           D_dict (Dictionary mit drei DTW-Distanzmatrizen)
# Funktionsweise: Berechnet für die übergebenen Zeitreihen paarweise DTW-Distanzen in drei Varianten: ohne Einschränkung, mit Sakoe-Chiba-Band und mit Itakura-Parallelogramm.
def getDTW(X_ts):
    # Standard DTW
    print(" [Info] Berechne DTW Standard")
    D_stand = cdist_dtw(X_ts)

    # DTW mit Sakoe-Chiba Band
    print(" [Info] Berechne DTW mit Sakoe-Chiba Band")
    D_sakoe = cdist_dtw(X_ts, sakoe_chiba_radius=10)

    # DTW mit Itakura Parallelogramm
    print(" [Info] Berechne DTW mit Itakura Parallelogramm")
    D_itakura = cdist_dtw(X_ts, itakura_max_slope=2.0)

    # Dictionary der drei DTW-Matritzen
    D_dict = {"standard": D_stand, "sakoe": D_sakoe, "itakura": D_itakura}

    return(D_dict)

# Funktion:         Transformiert eine Distanzmatrix in eine Ähnlichkeitsmatrix
# Input:            D (Distanzmatrix)
# Output:           A (Ähnlichkeitsmatrix)
# Funktionsweise:   Für die Tranformation der Distanzmatrix in eine Ähnlichkeitsmatrix wird der Gauß-Kernel (RBF) [s_ij = exp(-d_ij**2/2*sigma**2), wobei sigma = Median(d_ij) für d_ij > 0 ] verwendet. 
def dtw_transformation(D):
    sigma = np.median(D[D > 0])  # sigma = Median(d_ij) für d_ij > 0
    A = np.exp(-(D**2) / (2 * sigma**2)) # s_ij = exp(-d_ij**2/2*sigma**2)
    return A



# ---------- MODELLE ----------

# Funktion:         Erstellung eines Agglomerative-Clustering-Modells
# Input:            k (Anzahl der Cluster)
# Output:           model_agglomerative (Agglomerative-Clustering-Modell)
# Funktionsweise:   Erzeugt ein Agglomerative-Clustering-Modell mit der angegebenen Clusteranzahl
def model_agglomerative(k, linkage):
    model_agglomerative = AgglomerativeClustering(
        n_clusters=k,
        metric="precomputed",
        linkage=linkage
    )
    return(model_agglomerative)

# Funktion:         Erstellung eines Spectral-Clustering-Modells
# Input:            k (Anzahl der Cluster)
# Output:           model_spectral (Spectral-Clustering-Modell)
# Funktionsweise:   Erzeugt ein Spectral-Clustering-Modell mit der angegebenen Clusteranzahl
def model_spectral(k, assign_label):
    model_spectral = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        assign_labels=assign_label,
        random_state=123
    )
    return(model_spectral)

# Funktion:         Erstellung eines K-Medoids-Modells
# Input:            k (Anzahl der Cluster)
# Output:           model_kmedoids (K-Medoids-Modell)
# Funktionsweise:   Erzeugt ein K-Medoids-Modell mit der angegebenen Clusteranzahl
# def model_kmedoids(k):
#     model_kmedoids = KMedoids(
#         n_clusters=k,
#         metric="precomputed",
#         init="k-medoids++",
#         random_state=123)
#     return(model_kmedoids)
def model_kmedoids(k, init):
    model_kmedoids = KMedoids(
        n_clusters=k,
        metric="precomputed",
        init=init,
        random_state=123)
    return(model_kmedoids)

# Funktion:         Erstellung eines DBSCAN Modells
# Input:            eps_stand (Epsilon)
#                   min_samples_var (Mindestanzahl)
# Output:           model_dbscan (DBSCAN Modell)
# Funktionsweise:   Erzeugt ein DBSCAN-Modell mit der angegebenen Epsilon und Mindestanzahl
def model_dbscan(eps_stand, min_samples_var):
    model_dbscan = DBSCAN(
        eps=eps_stand,
        min_samples=min_samples_var, 
        metric="precomputed")
    return(model_dbscan)

# Funktion:         Erstellt ein Fuzzy C-Medoids auf Basis einer Distanzmatrix
# Input:            D (Distanzmatrix)
#                   n_clusters (Anzahl der Cluster)
#                   m (Fuzziness-Exponent)
#                   max_iteration (maximale Iterationsanzahl)
#                   konvergenz (Konvergenz-Schwellenwert)
# Output:           U (Zugehörigkeitsmatrix)
# Funktionsweise:   Zunächst wird eine zufällige Zugehörigkeitsmatrix initialisiert und iterativ optimiert.
#                   Dazu werden iterativ die Cluster-Zentretn aktualisiert und die Zugehörigkeiten aktualisiert.
#                   Nach dem erreichen des Konvergenz-Schwellenwert oder der maximale Iterationsanzahl wird der Algo abgebrochen. 
def model_fuzzy(D, n_clusters=10, m=2, max_iteration=100, konvergenz=1e-5):
    n = D.shape[0] # Anzahl der Zeitreihe
    np.random.seed(123) # Reproduzierbarkeit
    U = np.random.rand(n, n_clusters) # Zugehörigkeitsmatrix initialisieren
    U = U / U.sum(axis=1, keepdims=True) # Normierung: jede Zeile summiert auf 1

    for n in range(max_iteration):
        U_old = U.copy()

        # Cluster-Zentren bestimmen 
        centers_idx = []
        for c in range(n_clusters):
            weights = U[:, c] ** m
            costs = D.dot(weights)
            centers_idx.append(np.argmin(costs))
        centers_idx = np.array(centers_idx)

        # Cluster-Zugehörigkeiten bestimmen
        for i in range(n):
            d_i = D[i, centers_idx]
            d_i = np.maximum(d_i, 1e-10) # Division durch 0 verhindern
            for c in range(n_clusters):
                U[i, c] = 1.0 / np.sum((d_i[c] / d_i) ** (2/(m-1)))

        # Abbruchbedingung prüfen
        differenz = np.linalg.norm(U - U_old)
        if differenz < konvergenz:
            break

    return(U)
# In Anlehnung an: Dias, M. fuzzy-c-means. Abgerufen am 17.03.2026 von https://github.com/omadson/fuzzy-c-means/blob/master/fcmeans/main.py
# TODO: NEU !!!

# ---------- MODELL-HILFSFUNKTIONEN ----------
# Funktion:         Bestimmung des Dunn-Index
# Input:            D (Distanzmatrix)
#                   labels (Cluster-Zuordnung)
# Output:           dunn_index (Dunn-Index)
# Funktionsweise:   Bestimmung des Dunn-Index gemäß DI = min(δ(C_i, C_j)) / max(Δ(C_l)) 
def dunn_index_dtw(D, labels):
    clusters = [np.where(labels == l)[0] for l in np.unique(labels)]
    k = len(clusters)
        
    # min(δ(C_i, C_j)) (Abstand zwischen zwei verschiedenen Clustern)
    inter_dist = []
    for i in range(k):
        for j in range(i+1, k):
            dist_ij = D[np.ix_(clusters[i], clusters[j])]
            inter_dist.append(np.min(dist_ij))
    min_delta = min(inter_dist)

    # max(Δ(C_l))
    delta = []
    for cluster in clusters:
        if len(cluster) <= 1:
            delta.append(0)
            continue
        subD = D[np.ix_(cluster, cluster)]
        delta.append(np.max(subD))
    max_delta = max(delta)
    
    # Dunn-Index
    dunn_index = min_delta / max_delta

    return dunn_index

# Funktion:         Berechnet den Davies-Bouldin-Index
# Input:            D (Distanzmatrix)
#                   labels (Cluster-Zuordnung)
# Output:           davies_bouldin_index (Davies-Bouldin-Index )
# Funktionsweise:   Bestimmung des Davies-Bouldin-Index gemäß DBI = (1/k) * Σ max(R_ij) mit R_ij = (S_i + S_j) / M_ij
def davies_bouldin_dtw(D, labels):
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    k = len(unique_labels)
    
    # Cluster-Indices
    clusters = [np.where(labels == lab)[0] for lab in unique_labels]
    
    # S_i:
    S = []
    for cluster in clusters:
        if len(cluster) == 1:
            S.append(0)
            continue        
        intra_dist = D[np.ix_(cluster, cluster)]
        S_i = np.mean(intra_dist)
        S.append(S_i)
    
    S = np.array(S)
    
    # M_ij
    M = np.zeros((k, k))
    
    for i in range(k):
        for j in range(k):
            if i != j:
                dist_ij = D[np.ix_(clusters[i], clusters[j])]
                M[i, j] = np.mean(dist_ij)
    
    # R_ij = (S_i + S_j) / M_ij
    R = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i != j:
                R[i, j] = (S[i] + S[j]) / M[i, j]
    
    #DBI = (1/k) * Σ max(R_ij)
    max_R = np.max(R, axis=1)
    davies_bouldin_index = np.mean(max_R)
    
    return davies_bouldin_index


# Funktion:         Evaluation der Clusterergebnisse
# Input:            D (Distanzmatrix), 
#                   labels (Cluster-Zuordnung)
# Output:           sil_score (Silhouette-Score)
#                   dbi_score (Davies-Bouldin-Index)
#                   dunn_index_score (Dunn-Index)
# Funktionsweise:   Abhängig von den übergebendenen Labels und Distanzmatrix wird der Silhoutten-Score, Davies-Bouldin-Index und der Dunn-Index bestimmt und zurückgegeben.
def evaluate_cluster(D, labels):

    if len(set(labels)) < 2:
        return np.nan, np.nan, np.nan
    
    sil_score = round(silhouette_score(D, labels, metric="precomputed"),2)
    dbi_score = round(davies_bouldin_dtw(D, labels),2)
    dunn_index_score = round(dunn_index_dtw(D, labels),2)

    return sil_score, dbi_score, dunn_index_score


# Für jede DTW-Variante den optimalen eps berechnen
# Funktion:         Bestimmt den optimalen Epsilon-Wert
# Input:            D (Distanzmatrix)
#                   k (Anzahl der Nachbarn)
# Output:           eps (Epsilon)
# Funktionsweise:   Für jeden Punkt wird die Distanz zum k-nöchsten Nachbarn mittels NearestNeighbors, basierend auf der Distanzmatrix bestimmt,
#                   Anschließend werden die k-Distanzen aufsteigend sortiert. Der Kniepunkt der Kurve wird mit dem KneeLocators ermittelt und als optimaler Epsilon-Wert zurpckgegeben.
def find_optimal_eps(D, k):
    model_nn = NearestNeighbors(n_neighbors=k, metric="precomputed")
    distances, _ = model_nn.fit(D).kneighbors(D)
    k_distances = np.sort(distances[:, k-1])
    
    knee = KneeLocator(
        range(len(k_distances)), 
        k_distances, 
        curve="convex", 
        direction="increasing"
    )

    eps = k_distances[knee.knee]
    return eps


# Funktion:         Bestimmung der optimalen Clusteranzahl
# Input:            D (Distanzmatrix)
#                   model_name (Names des Modells)
# Output:           optimal_k (optimale Clusteranzahl)
# Funktionsweise:   Es werden für das übergebende Modell verschiedene Clusteranzahlen k getestet.
#                   Dazu wird jedes k auf das Modell angewendet und dafür den Silhouette-Score bestimmt. 
#                   Das k mit dem höchsten Score, wird als optimales k zurückgegeben. 
def find_opt_params(D, model_name, A = None): # TODO
    list_k = [2, 3, 4, 5, 6, 7,8, 9, 10] 

    best_score = -1
    best_params = {}

    if(model_name == "kmedoids"):
        for k in list_k:
            for init in ["k-medoids++", "heuristic", "random"]:
                labels = model_kmedoids(k, init).fit_predict(D)
                score = silhouette_score(D, labels, metric="precomputed")
                if score > best_score:
                    best_score = score
                    best_params = {"k": k, "init": init}

    if model_name == "agglomerative":
        for k in list_k:
            for linkage in ["average", "complete", "single"]:
                labels = model_agglomerative(k, linkage).fit_predict(D)
                score = silhouette_score(D, labels, metric="precomputed")
                if score > best_score:
                    best_score = score
                    best_params = {"k": k, "linkage": linkage}

    if model_name == "spectral":
        for k in list_k:
            for assign_label in ["kmeans", "discretize"]:
                labels = model_spectral(k, assign_label).fit_predict(A)
                score = silhouette_score(D, labels, metric="precomputed")
                if score > best_score:
                    best_score = score
                    best_params = {"k": k, "assign_label": assign_label}

    if(model_name == "fuzzy"):
        for k in list_k:
            for m in [1.5, 2.0, 2.5, 3.0]:
                U = model_fuzzy(D, n_clusters=k, m=m)
                labels = np.argmax(U, axis=1)
                score = silhouette_score(D, labels, metric="precomputed")
                if score > best_score:
                    best_score = score
                    best_params = {"k": k, "m": m}

    return best_params

# Funktion:         Wendet eine Clusteringmethode auf alle DTW-Varianten an und evaluiert die Ergebnisse
# Input:            clustering_model_name (Name der Clusteringmethode)
#                   clustering_model (Clusteringmodell)
#                   D_dict (Dictionary mit DTW-Distanzmatrizen)
#                   df_evaluate (Evaluationsergebnisse-DataFrame)
#                   n_run (Durchgangsnummer)
#                   X_scaled (skalierte Zeitreihen)
# Output:           dict_labels (Dictionary mit Cluster-Zuordnungen pro DTW-Variante)
# Funktionsweise:   Jedes Modell wird über alle DTW-Varianten Iteriert. 
#                   Dabei werden die optimalen Parameter bestimmt, die Modelle angewendet, die Evaluationskennzahlen bestimmt und im Evaluationsergebnisse-DataFrame gespeichert sowie die Cluster-Zuordnungen gespeichert.
def runModel(clustering_model_name, clustering_model, D_dict, df_evaluate, n_run, X_scaled = None):
    print(f"  [Info] Starte Clusteringmethode {clustering_model_name}")
    dict_labels = {}

    if(clustering_model_name == "kmedoids"):
        for D_kind, D in D_dict.items():
            best_params = find_opt_params(D, clustering_model_name) # Optimale Parameter bestimmen
            labels = model_kmedoids(best_params["k"], init=best_params["init"]).fit_predict(D) # Modell mit optimalen Parametern anwenden
            df_evaluate.loc[len(df_evaluate)] = [n_run, clustering_model_name, D_kind, best_params["k"], *evaluate_cluster(D, labels)] # Evaluationskennzahlen bestimmen
            dict_labels[D_kind] = labels 

    if(clustering_model_name == "agglomerative"):
        for D_kind, D in D_dict.items():
            best_params = find_opt_params(D, clustering_model_name) # Optimale Parameter bestimmen
            labels = model_agglomerative(best_params["k"], best_params["linkage"]).fit_predict(D) # Modell mit optimalen Parametern anwenden
            df_evaluate.loc[len(df_evaluate)] = [n_run, clustering_model_name, D_kind, best_params["k"], *evaluate_cluster(D, labels)] # Evaluationskennzahlen bestimmen
            dict_labels[D_kind] = labels 

    if(clustering_model_name == "spectral"):
        for D_kind, D in D_dict.items():
            A = dtw_transformation(D) # Affinitätsmatrix bestimmen
            best_params = find_opt_params(D, clustering_model_name, A) # Optimale Parameter bestimmen
            labels = model_spectral(best_params["k"], best_params["assign_label"]).fit_predict(A) # Modell mit optimalen Parametern anwenden
            df_evaluate.loc[len(df_evaluate)] = [n_run, clustering_model_name, D_kind, best_params["k"], *evaluate_cluster(D, labels)] # Evaluationskennzahlen bestimmen
            dict_labels[D_kind] = labels 

    if(clustering_model_name == "fuzzy"):
        for D_kind, D in D_dict.items():
            best_params = find_opt_params(D, clustering_model_name) # Optimale Parameter bestimmen
            U = model_fuzzy(D, n_clusters=best_params["k"], m=best_params["m"])
            labels = np.argmax(U, axis=1)
            n_clusters   = len(set(labels))
            df_evaluate.loc[len(df_evaluate)] = [n_run, clustering_model_name, D_kind, best_params["k"], *evaluate_cluster(D, labels)] # Evaluationskennzahlen bestimmen
            dict_labels[D_kind] = labels 

    if(clustering_model_name == "dbscan"):
        for D_kind, D in D_dict.items():
            # Optimale Parameter bestimmen
            min_samples_var = int(np.log(len(X_scaled)))  
            eps_stand   = round(find_optimal_eps(D, min_samples_var),0)
            labels = model_dbscan(eps_stand, min_samples_var).fit_predict(D)
            n_clusters   = len(set(labels))
            df_evaluate.loc[len(df_evaluate)] = [n_run, clustering_model_name, D_kind, n_clusters, *evaluate_cluster(D, labels)] # Evaluationskennzahlen bestimmen  
            dict_labels[D_kind] = labels 

    return(dict_labels)
