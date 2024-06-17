import os
import cv2 as cv
import numpy as np
from skimage.feature import hog
from skimage import transform
from skimage.feature import hog
import itertools
from PIL import Image


def compute_color_histograms(images):
    """
    Calcule les histogrammes de niveau de gris pour les images MNIST.
    Input : images (list) : liste des images en niveaux de gris
    Output : descriptors (list) : liste des descripteurs d'histogrammes de niveau de gris
    """
    descriptors = []
    for image in images:
        image = image.astype(np.uint8)
        hsv_image = cv.cvtColor(image, cv.COLOR_RGB2HSV)
        hist = cv.calcHist([hsv_image],[0,1,2],None,[16,16,16],[0,180,0,256,0,256])
        descriptors.append(cv.normalize(hist,hist).flatten())
    return descriptors

def compute_ORB_descriptor(images, targeted_feature = 300):
    descriptors = []
    
    # ORB
    orb = cv.ORB_create(targeted_feature)  # Créer un détecteur ORB
    for image in images:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # Convertir l'image en niveaux de gris
        keypoints = orb.detect(gray, None)  # Détecter les points d'intérêt ORB
        kp, descriptor = orb.compute(gray, keypoints)  # Calculer les descripteurs ORB
        descriptors.append(descriptor)
    
    #on met tous les descripteurs à la même taile pour cela on cherche le plus petit et on tronque les autres
    descriptors_resized = []
    for descriptor in descriptors:
        if descriptor.shape[0] < targeted_feature:
            # Si le descripteur est plus petit que la taille désirée, le remplir avec des zéros
            padded_descriptor = np.zeros((targeted_feature,32), dtype=descriptor.dtype)
            padded_descriptor[:descriptor.shape[0]] = descriptor
            descriptors_resized.append(cv.normalize(padded_descriptor,padded_descriptor).flatten())
        else:
            descriptors_resized.append(cv.normalize(descriptor,descriptor).flatten())


    return descriptors_resized


def compute_hog_descriptors(images):
    """
    Calcule les descripteurs HOG pour les images en niveaux de gris.
    Input : images (array) : tableau numpy des images
    Output : descriptors (list) : liste des descripteurs HOG
    """
    descriptors = []
    i = 0
    for image in images:
        """ 
        
        """
        # Redimensionner l'image avec PIL
        pil_img = Image.fromarray(image)
        resized_img = pil_img.resize((150, 150))
        resized_img_array = np.array(resized_img)
        #
        gray = cv.cvtColor(resized_img_array , cv.COLOR_BGR2GRAY)
        # Compute HOG descriptors

        fd, hog_image = hog(gray , orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualize=True)
        
        descriptors.append(cv.normalize(fd,fd).flatten())
        
    return descriptors