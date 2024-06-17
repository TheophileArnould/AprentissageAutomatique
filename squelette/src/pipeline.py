from sklearn.preprocessing import StandardScaler
import os
import pandas as pd

import glob
import numpy as np
import plotly.express as px
import cv2 as cv



from features import *
from clustering import *
from utils import *
from constant import PATH_OUTPUT,PATH_DATA, MODEL_CLUSTERING



def pipeline():
   
    # Récupération des données
    print("\n\n ##### Récupération des données ######")
    path = glob.glob(PATH_DATA + '/*/*.jpg')
    labelImages = []
    for img in path:
        n = cv.imread(img)
        if(n is not None):
            n = np.array(n, dtype=np.uint8)
            n = cv.cvtColor(n,cv.COLOR_BGR2RGB)
            imgSplit = img.split('/')

            labelImages.append([imgSplit[4],n])
        
    labelImages = pd.DataFrame(labelImages, columns=['Label', 'Image'])
    images = labelImages['Image']
    images = images.to_numpy()

    Labels_true = labelImages['Label']
    labels_true = Labels_true.to_numpy()

    print(str(len(images)) + " images ont été récupérées")


    print("\n\n ##### Extraction de Features ######")
    print("- calcul features Color...")
    # TODO
    descriptors_color_hist = compute_color_histograms(images)

    print("- calcul features ORB...")
    descriptors_ORB = compute_ORB_descriptor(images,200)

    print("- calcul features HOG...")
    descriptors_HOG = compute_hog_descriptors(images)



    descriptors_ORB_COLOR_HOG = np.concatenate((descriptors_color_hist, descriptors_ORB, descriptors_HOG), axis=1)

    descriptors_ORB_HOG = np.concatenate((descriptors_color_hist, descriptors_ORB, descriptors_HOG), axis=1)

    descriptors_COLOR_COLOR = np.concatenate((descriptors_color_hist, descriptors_color_hist), axis = 1)
    print("\n\n ##### Clustering ######")
    number_cluster = 20

    print("- calcul SpectralClustering avec features color, orb, et hog et " + str(number_cluster) + " clusters  ...")
    # TODO

    #COLOR
    spectral_color = SpectralClustering(n_clusters=number_cluster)
    spectral_color.fit(np.array(descriptors_color_hist))

    #ORB_COLOR_HOG
    spectral_orb_color_hog = SpectralClustering(n_clusters=number_cluster)
    spectral_orb_color_hog.fit(np.array(descriptors_ORB_COLOR_HOG))

    #ORB_HOG
    spectral_orb_hog = SpectralClustering(n_clusters=number_cluster)
    spectral_orb_hog.fit(np.array(descriptors_ORB_HOG))

    print("- calcul DBSCAN avec features color, orb et hog et " + str(number_cluster) + " clusters  ...")

    #COLOR
    DBSCAN_color = DBSCAN(eps = 1.158, min_samples = 5)
    DBSCAN_color.fit(np.array(descriptors_COLOR_COLOR))

    # ORB & COLOR & HOG
    DBSCAN_orb_color_hog = DBSCAN(eps = 1.158, min_samples = 3)
    DBSCAN_orb_color_hog.fit(np.array(descriptors_ORB_COLOR_HOG))

    #ORB&HOG
    DBSCAN_orb_hog = DBSCAN(eps = 0.905, min_samples = 3)
    DBSCAN_orb_hog.fit(np.array(descriptors_ORB_HOG))


    print("\n\n ##### Résultat ######")
    #SPECTRAL
    metric_color_Spec = show_metric(labels_true,spectral_color.labels_, descriptors_color_hist, bool_show=True, name_model="SpectralClustering" , name_descriptor="Color", bool_return=True)
    metric_orb_color_hog_Spec = show_metric(labels_true, spectral_orb_color_hog.labels_, descriptors_ORB_COLOR_HOG,bool_show=True, name_model="SpectralClustering" , name_descriptor="ORB_COLOR_HOG", bool_return=True)
    metric_orb_hog_Spec = show_metric(labels_true, spectral_orb_hog.labels_, descriptors_ORB_HOG,bool_show=True, name_model="SpectralClustering" , name_descriptor="ORB_HOG", bool_return=True)

    #DBSCAN
    metric_color_DB = show_metric(labels_true,DBSCAN_color.labels_, descriptors_color_hist, bool_show=True , name_model="DBSCAN" , name_descriptor="Color", bool_return=True)
    metric_orb_color_hog_DB = show_metric(labels_true, DBSCAN_orb_color_hog.labels_, descriptors_ORB_COLOR_HOG,bool_show=True, name_model="DBSCAN" , name_descriptor="ORB_COLOR", bool_return=True)
    metric_orb_hog_DB = show_metric(labels_true, DBSCAN_orb_hog.labels_, descriptors_ORB_HOG,bool_show=True, name_model="DBSCAN" , name_descriptor="ORB", bool_return=True)

    print("- export des données vers le dashboard")
    # conversion des données vers le format du dashboard
    list_dict = [metric_color_Spec,metric_orb_color_hog_Spec, metric_orb_hog_Spec, metric_color_DB,metric_orb_color_hog_DB , metric_orb_hog_DB]
    df_metric = pd.DataFrame(list_dict)
    
    # Normalisation des données
    scaler = StandardScaler()
    descriptors_color_hist_norm = scaler.fit_transform(descriptors_color_hist)
    descriptors_orb_color_hog_norm = scaler.fit_transform(descriptors_ORB_COLOR_HOG)
    descriptors_orb_hog_norm = scaler.fit_transform(descriptors_ORB_HOG)

    #conversion vers un format 3D pour la visualisation
    print("- conversion vers le format 3D ...")
    x_3d_color = conversion_3d(descriptors_color_hist_norm)
    x_3d_orb_color_hog = conversion_3d(descriptors_orb_color_hog_norm)
    x_3d_orb_hog = conversion_3d(descriptors_orb_hog_norm)

    # création des dataframe pour la sauvegarde des données pour la visualisation
    df_Spectral_color = create_df_to_export(x_3d_color, labels_true, spectral_color.labels_)
    df_Spectral_orb_color_hog = create_df_to_export(x_3d_orb_color_hog, labels_true, spectral_orb_color_hog.labels_)
    df_Spectral_orb_hog = create_df_to_export(x_3d_orb_hog, labels_true, spectral_orb_hog.labels_)

    df_DBSCAN_color = create_df_to_export(x_3d_color, labels_true, DBSCAN_color.labels_)
    df_DBSCAN_orb_color_hog = create_df_to_export(x_3d_orb_color_hog, labels_true, DBSCAN_orb_color_hog.labels_)
    df_DBSCAN_orb_hog = create_df_to_export(x_3d_orb_hog, labels_true, DBSCAN_orb_hog.labels_)



    # Vérifie si le dossier existe déjà
    if not os.path.exists(PATH_OUTPUT):
        # Crée le dossier
        os.makedirs(PATH_OUTPUT)

    # sauvegarde des données
    print("- enregistrement des fichiers ...")
    df_Spectral_color.to_excel(PATH_OUTPUT+"/clustering_color_Spectral.xlsx")
    df_Spectral_orb_color_hog.to_excel(PATH_OUTPUT+"/clustering_orb_color_hog_Spectral.xlsx")
    df_Spectral_orb_hog.to_excel(PATH_OUTPUT+"/clustering_orb_hog_Spectral.xlsx")

    df_DBSCAN_color.to_excel(PATH_OUTPUT+"/clustering_color_DBSCAN.xlsx")
    df_DBSCAN_orb_color_hog.to_excel(PATH_OUTPUT+"/clustering_orb_color_hog_DBSCAN.xlsx")
    df_DBSCAN_orb_hog.to_excel(PATH_OUTPUT+"/clustering_orb_hog_DBSCAN.xlsx")


    df_metric.to_excel(PATH_OUTPUT+"/save_metric.xlsx")
    print("Fin. \n\n Pour avoir la visualisation dashboard, veuillez lancer la commande : streamlit run dashboard_clustering.py")


if __name__ == "__main__":
    pipeline()