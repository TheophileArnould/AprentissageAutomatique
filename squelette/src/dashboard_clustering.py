import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import plotly.express as px
import cv2 as cv
import glob
from constant import PATH_OUTPUT,PATH_DATA, MODEL_CLUSTERING



@st.cache_data
def colorize_cluster(cluster_data, selected_cluster):
    fig = px.scatter_3d(cluster_data, x='x', y='y', z='z', color='cluster')
    filtered_data = cluster_data[cluster_data['cluster'] == selected_cluster]
    fig.add_scatter3d(x=filtered_data['x'], y=filtered_data['y'], z=filtered_data['z'],
                    mode='markers', marker=dict(color='red', size=10),
                    name=f'Cluster {selected_cluster}')
    return fig

        
# Chargement des données du clustering
df_color_DB = pd.read_excel("../../output/clustering_color_DBSCAN.xlsx")
df_color_Spectral = pd.read_excel("../../output/clustering_color_Spectral.xlsx")
df_orb_color_hog_DB = pd.read_excel("../../output/clustering_orb_color_hog_DBSCAN.xlsx")
df_orb_color_hog_spectral = pd.read_excel("../../output/clustering_orb_color_hog_Spectral.xlsx")
df_orb_hog_spectral = pd.read_excel("../../output/clustering_orb_hog_Spectral.xlsx")
df_orb_hog_DB = pd.read_excel("../../output/clustering_orb_hog_DBSCAN.xlsx")


df_metric = pd.read_excel("../../output/save_metric.xlsx")

if 'Unnamed: 0' in df_metric.columns:
    df_metric.drop(columns="Unnamed: 0", inplace=True)

# Création de deux onglets
tab1, tab2,tab3 = st.tabs(["Analyse par descripteur", "Analyse global","Echantillon d'image" ])

# Onglet numéro 1
with tab1:

    st.write('## Résultat de Clustering des données FRUITS')
    st.sidebar.write("####  Veuillez sélectionner les clusters à analyser" )
    # Sélection des descripteurs
    descriptor =  st.sidebar.selectbox('Sélectionner un descripteur', ["COLOR","ORB HOG","ORB HOG COLOR"])
    algo =  st.sidebar.selectbox('Sélectionner un Algotithme de clustering', ["BDSCAN","Spectral"])
    if descriptor=="COLOR":
        if(algo == "BDSCAN"):
            df = df_color_DB
        else:
            df = df_color_Spectral
    elif descriptor=="ORB HOG":
        if(algo == "BDSCAN"):
            df = df_orb_hog_DB
        else:
            df = df_orb_hog_spectral
    elif descriptor=="ORB HOG COLOR":
        if(algo == "BDSCAN"):
            df = df_orb_color_hog_DB
        else:
            df = df_orb_color_hog_spectral

    # Ajouter un sélecteur pour les clusters
    selected_cluster =  st.sidebar.selectbox('Sélectionner un Cluster', range(len(df['cluster'].unique())))
    # Filtrer les données en fonction du cluster sélectionné
    cluster_indices = df[df.cluster==selected_cluster].index    
    st.write(f"###  Analyse du descripteur {descriptor}" )
    st.write(f"#### Analyse du cluster : {selected_cluster}")
    st.write(f"####  Visualisation 3D du clustering avec descripteur {descriptor}" )
    # Sélection du cluster choisi
    filtered_data = df[df['cluster'] == selected_cluster]
    # Création d'un graph 3D des clusters
    # TODO : à remplir
    fig = px.scatter_3d(filtered_data, x='x', y='y', z='z',color='label')

    st.plotly_chart(fig)

# Onglet numéro 2
with tab2:
    st.write('## Analyse Global des descripteurs' )

    global_metric_choice = st.selectbox('Select Global Metric for Analysis', ["ami","silhouette"])

    if global_metric_choice in df_metric.columns:
        fig = px.bar(df_metric[global_metric_choice])
        st.plotly_chart(fig)
