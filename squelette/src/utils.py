import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
from matplotlib import pyplot as plt 

def conversion_3d(X, n_components=3,perplexity=50,random_state=42, early_exaggeration=10,n_iter=3000):
    """
    Conversion des vecteurs de N dimensions vers une dimension précise (n_components) pour la visualisation
    Input : X (array-like) : données à convertir en 3D
            n_components (int) : nombre de dimensions cibles (par défaut : 3)
            perplexity (float) : valeur de perplexité pour t-SNE (par défaut : 50)
            random_state (int) : graine pour la génération de nombres aléatoires (par défaut : 42)
            early_exaggeration (float) : facteur d'exagération pour t-SNE (par défaut : 10)
            n_iter (int) : nombre d'itérations pour t-SNE (par défaut : 3000)
    Output : X_3d (array-like) : données converties en 3D
    """
    X_3d = []
    #TODO :  Conversion vers format 3D avec TSNE.
    X_3d = TSNE(n_components=n_components, learning_rate='auto',init='random', perplexity=perplexity, random_state=random_state, early_exaggeration=early_exaggeration, n_iter=n_iter).fit_transform(X)
    return X_3d


def create_df_to_export(data_3d, l_true_label,l_cluster):
    """
    Création d'un DataFrame pour stocker les données et les labels
    Input : data_3d (array-like) : données converties en 3D
            l_true_label (list) : liste des labels vrais
            l_cluster (list) : liste des labels de cluster
            l_path_img (list) : liste des chemins des images
    Output : df (DataFrame) : DataFrame contenant les données et les labels
    """
    df = pd.DataFrame(data_3d, columns=['x', 'y', 'z'])
    df['label'] = l_true_label
    df['cluster'] = l_cluster
    
    return df

def afficher_clusters(images, labels, max_image = 10):
    # Créer un dictionnaire pour regrouper les images par cluster
    clusters = {}
    for label, image in zip(labels, images):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(image)

    # Afficher chaque cluster
    for label, cluster_images in clusters.items():
        print("Cluster", label)
        if len(cluster_images) == 1:
            # Si le cluster contient une seule image, utiliser une liste pour stocker l'objet Axes
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(cluster_images[0])
            ax.axis('off')
            plt.show()
        
        elif(len(cluster_images) < max_image):
            # Sinon, utiliser subplots normalement
            fig, axes = plt.subplots(1, len(cluster_images), figsize=(15, 15))
            for i, image in enumerate(cluster_images):
                
                axes[i].imshow(image)
                axes[i].axis('off')
            plt.show()
        else:
            # Sinon, utiliser subplots limité à 10
            fig, axes = plt.subplots(1, max_image, figsize=(15, 15))
            for i, image in enumerate(cluster_images):
                if(i == max_image):
                    break
                axes[i].imshow(image)
                axes[i].axis('off')
            plt.show()
            
