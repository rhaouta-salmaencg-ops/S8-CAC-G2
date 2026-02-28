python
# -*- coding: utf-8 -*-
"""Formation ML & DL : S2 Clustering

<center><h1>Formation en Machine Learning et Deep Learning</h1></center>
<center><h3>Atelier 2 : Comprendre l’Apprentissage Non Supervisé</h3></center>

**Objectif :**
Initier les apprenants à l’apprentissage non supervisé.
Nous allons utiliser les principaux algorithmes suivants :

• K-Means : clustering partitionnel  
• DBSCAN : clustering basé sur la densité  
• Clustering hiérarchique : approche agglomérative  
• Modèle de Mélange de Gaussiennes (GMM - Gaussian Mixture Model)

# Introduction

L’apprentissage non supervisé est une branche du Machine Learning dont l’objectif est d’analyser et de structurer des données sans étiquettes prédéfinies (labels).

Contrairement à l’apprentissage supervisé, les données ne contiennent pas d’étiquettes. Les algorithmes se basent uniquement sur les similarités et les structures présentes dans les données pour regrouper les observations.

Dans cet atelier, nous allons explorer les méthodes de clustering, parmi les plus utilisées en apprentissage non supervisé.

Bibliothèques :
• Scikit-learn : bibliothèque pour l’apprentissage automatique  
• Matplotlib et Seaborn : outils de visualisation  
"""
Importation des bibliothèques
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
📌Chargement du Dataset Iris
from sklearn.datasets import load_iris

dataset = load_iris()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

display(df.head())
📌 Visualisation des données
def plot_initial_data():
    plt.figure(figsize=(6,6))
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], s=50, alpha=0.7)
    plt.title("Données initiales")
    plt.xlabel(dataset.feature_names[0])
    plt.ylabel(dataset.feature_names[1])
    plt.show()

plot_initial_data()
🔵 K-MEANS
from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=2,
    init="k-means++",
    max_iter=100,
    random_state=42
)

kmeans_labels = kmeans.fit_predict(df)
Visualisation
def plot_clusters(labels, title):
    plt.figure(figsize=(6,6))
    sns.scatterplot(
        x=df.iloc[:,0],
        y=df.iloc[:,1],
        hue=labels,
        palette="viridis",
        s=50
    )
    plt.title(title)
    plt.xlabel(dataset.feature_names[0])
    plt.ylabel(dataset.feature_names[1])
    plt.show()

plot_clusters(kmeans_labels, "K-Means Clustering")
📌 Méthode du coude
inertia = []

for k in range(1, 10):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(df)
    inertia.append(model.inertia_)

plt.plot(range(1,10), inertia, marker="o")
plt.xlabel("Nombre de clusters")
plt.ylabel("Inertie")
plt.title("Méthode du coude")
plt.show()
🔵 DBSCAN
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(
    eps=0.5,
    min_samples=5,
    metric="euclidean"
)

dbscan_labels = dbscan.fit_predict(df)

plot_clusters(dbscan_labels, "DBSCAN Clustering")
🔵 Clustering Hiérarchique
from scipy.cluster.hierarchy import dendrogram, linkage

linked = linkage(df, method="ward")

plt.figure(figsize=(8,5))
dendrogram(linked)
plt.title("Dendrogramme")
plt.show()
from sklearn.cluster import AgglomerativeClustering

agglo = AgglomerativeClustering(
    n_clusters=2,
    metric="euclidean",
    linkage="ward"
)

agglo_labels = agglo.fit_predict(df)

plot_clusters(agglo_labels, "Agglomerative Clustering")
🔵 GMM
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(
    n_components=2,
    covariance_type="full",
    max_iter=100,
    random_state=42
)

gmm_labels = gmm.fit_predict(df)

plot_clusters(gmm_labels, "GMM Clustering")
🔵 Évaluation : Silhouette Score
from sklearn.metrics import silhouette_score

def evaluate_clusters(labels, method):
    score = silhouette_score(df, labels)
    print(f"Silhouette Score ({method}) : {score:.2f}")

evaluate_clusters(kmeans_labels, "K-Means")
evaluate_clusters(dbscan_labels, "DBSCAN")
evaluate_clusters(agglo_labels, "Agglomerative")
evaluate_clusters(gmm_labels, "GMM")
🖼 SEGMENTATION D’IMAGE
📌 Chargement Image
import cv2

image_path = "/content/Feuille.bmp"
image = cv2.imread(image_path)

# Conversion BGR → RGB (correction importante)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.axis("off")
plt.show()
📌 Segmentation niveau de gris
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

pixels = image_gray.reshape(-1,1)

kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(pixels)

segmented = labels.reshape(image_gray.shape)

binary_image = (segmented * 255).astype(np.uint8)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(image_rgb)
plt.title("Image originale")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(binary_image, cmap="gray")
plt.title("Segmentation K-Means")
plt.axis("off")

plt.show()
📌 Segmentation RGB
pixels_rgb = image_rgb.reshape(-1,3)

kmeans = KMeans(n_clusters=2, random_state=42)
labels_rgb = kmeans.fit_predict(pixels_rgb)

segmented_rgb = labels_rgb.reshape(image_rgb.shape[:2])

binary_rgb = (segmented_rgb * 255).astype(np.uint8)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(image_rgb)
plt.title("Image originale")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(binary_rgb, cmap="gray")
plt.title("Segmentation RGB")
plt.axis("off")

plt.show()
📌 Segmentation par canal R, G, B
R, G, B = cv2.split(image_rgb)

def segment_channel(channel):
    pixels = channel.reshape(-1,1)
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(pixels)
    return (labels.reshape(channel.shape) * 255).astype(np.uint8)

segR = segment_channel(R)
segG = segment_channel(G)
segB = segment_channel(B)

segmented_imageRGB = cv2.merge([segR, segG, segB])

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(image_rgb)
plt.title("Image originale")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(segmented_imageRGB)
plt.title("Segmentation par canal RGB")
plt.axis("off")

plt.show()
