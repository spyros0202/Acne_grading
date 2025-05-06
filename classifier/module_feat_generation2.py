# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 18:19:02 2022

@author: medisp-2
"""

#module module_feat_generation
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
# https://github.com/Rohit-Kundu/Traditional-Feature-Extraction   !!!!!!!!!!!


def generate_local_binary_pattern_featuresLBP4(im):
    # https://scikit-image.org/docs/0.7.0/api/skimage.feature.texture.html#skimage.feature.texture.greycoprops
    # https://pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
    # https://fairyonice.github.io/implement-lbp-from%20scratch.html
    # https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_local_binary_pattern.html
    import skimage
    from skimage import feature
    R = 1;
    P = 8  # (Number of points =16 and radius =2)

    lbp = feature.local_binary_pattern(im, P, R, method='uniform')
    # print(np.min(lbp));print(np.max(lbp))

    # plt.imshow(np.uint8(lbp),[])
    # U.RETURN()
    lbp = np.asarray(lbp)
    # fz=8
    # plt.figure(figsize=(fz*1.5,fz));
    # plt.subplot(1,2,1)
    # plt.imshow(255*(lbp/np.max(lbp)),cmap='gray')
    # plt.title('LBP-image')
    lbp_fl = lbp.ravel()
    hist1, _ = np.histogram(lbp_fl, bins=np.arange(0, P + 3), range=(0, P + 2))
    hist1 = hist1.astype("float")
    # plt.subplot(1,2,2)
    # plt.hist(lbp_fl,bins=np.arange(0,P+3),range=(0,P+2),edgecolor='black',linewidth=1.2)
    # plt.grid();plt.title('LBP histogram');plt.xlabel('bins');plt.ylabel('frequency')
    # plt.show()
    fNs = [];
    features = []

    fNs.append('LBP_4')
    features.append(hist1[3])
    fNs = np.asarray(fNs, str)
    features = np.asarray(features, float)

    return (fNs, features)

def generate_local_binary_pattern_featuresLBP1(im):
    # https://scikit-image.org/docs/0.7.0/api/skimage.feature.texture.html#skimage.feature.texture.greycoprops
    # https://pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
    # https://fairyonice.github.io/implement-lbp-from%20scratch.html
    # https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_local_binary_pattern.html
    import skimage
    from skimage import feature
    R = 1;
    P = 8  # (Number of points =16 and radius =2)

    lbp = feature.local_binary_pattern(im, P, R, method='uniform')
    # print(np.min(lbp));print(np.max(lbp))

    # plt.imshow(np.uint8(lbp),[])
    # U.RETURN()
    lbp = np.asarray(lbp)
    # fz=8
    # plt.figure(figsize=(fz*1.5,fz));
    # plt.subplot(1,2,1)
    # plt.imshow(255*(lbp/np.max(lbp)),cmap='gray')
    # plt.title('LBP-image')
    lbp_fl = lbp.ravel()
    hist1, _ = np.histogram(lbp_fl, bins=np.arange(0, P + 3), range=(0, P + 2))
    hist1 = hist1.astype("float")
    # plt.subplot(1,2,2)
    # plt.hist(lbp_fl,bins=np.arange(0,P+3),range=(0,P+2),edgecolor='black',linewidth=1.2)
    # plt.grid();plt.title('LBP histogram');plt.xlabel('bins');plt.ylabel('frequency')
    # plt.show()
    fNs = [];
    features = []

    fNs.append('LBP_1')
    features.append(hist1[0])
    fNs = np.asarray(fNs, str)
    features = np.asarray(features, float)

    return (fNs, features)


def generate_2ndOrder_coocurrence_matrix_features1(im):
    # https://scikit-image.org/docs/0.7.0/api/skimage.feature.texture.html
    levs = 15
    im = levs * (im / np.max(im))
    image = np.asarray(im, dtype=np.uint8)

    # print(image)

    import skimage
    from skimage.feature import graycomatrix, graycoprops
    result = graycomatrix(image, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=levs + 1)
    # print("------1--------")
    # print (result[:,:,0,0])
    # print("------2--------")
    # print (result[:,:,0,1])
    # print("------3--------")
    # print (result[:,:,0,2])
    # print("------4--------")
    # print (result[:,:,0,3])
    # print (result)

    # https://scikit-image.org/docs/0.7.0/api/skimage.feature.texture.html#skimage.feature.texture.graycoprops

    correlation = graycoprops(result, prop='correlation')


    features = np.zeros(1)
    feature_names = [ 'correlation_range',

                     ]

    correlation_range = \
        np.max(correlation) - np.min(correlation)


    features[0] = correlation_range

    # print(features)
    return (feature_names, features)


def generate_2ndOrder_coocurrence_matrix_features2(im):
    # https://scikit-image.org/docs/0.7.0/api/skimage.feature.texture.html
    levs = 15
    im = levs * (im / np.max(im))
    image = np.asarray(im, dtype=np.uint8)

    # print(image)

    import skimage
    from skimage.feature import graycomatrix, graycoprops
    result = graycomatrix(image, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=levs + 1)
    # print("------1--------")
    # print (result[:,:,0,0])
    # print("------2--------")
    # print (result[:,:,0,1])
    # print("------3--------")
    # print (result[:,:,0,2])
    # print("------4--------")
    # print (result[:,:,0,3])
    # print (result)

    # https://scikit-image.org/docs/0.7.0/api/skimage.feature.texture.html#skimage.feature.texture.graycoprops

    dis = graycoprops(result, prop='dissimilarity')


    features = np.zeros(1)
    feature_names = [
                     'dis_mean'

                     ]
    dis_mean = np.mean(dis);



    features[0] = dis_mean;

    # print(features)
    return (feature_names, features)