# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 18:19:02 2022

@author: medisp-2
"""

# module module_feat_generation
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import moduleUtils as U
import cv2


def apply_over_degree(function, x1, x2):
    rows, cols, nums = x1.shape
    result = np.ndarray((rows, cols, nums))
    for i in range(nums):
        # print(x1[:, :, i])
        result[:, :, i] = function(x1[:, :, i], x2)
        # print(result[:, :, i])
        result[result == np.inf] = 0
        result[np.isnan(result)] = 0
    return result


# ----------------------------------------------------------------
def calcuteIJ(rlmatrix):
    gray_level, run_length, _ = rlmatrix.shape
    I, J = np.ogrid[0:gray_level, 0:run_length]
    return I, J + 1


def calcuteS(rlmatrix):
    return np.apply_over_axes(np.sum, rlmatrix, axes=(0, 1))[0, 0]


# The following code realizes the extraction of 11 gray runoff matrix features



# 2.LRE
def getLongRunEmphasis(rlmatrix):
    I, J = calcuteIJ(rlmatrix)
    numerator = np.apply_over_axes(np.sum, apply_over_degree(np.multiply, rlmatrix, (J * J)), axes=(0, 1))[0, 0]
    S = calcuteS(rlmatrix)
    return numerator / S

def generate_RL_matrix(array, theta):
    from itertools import groupby
    # ******** from https://github.com/Rohit-Kundu/Traditional-Feature-Extraction ************
    #array: the numpy array of the image
    #theta: Input, the angle used when calculating the gray scale run matrix, list type, can contain fields:['deg0', 'deg45', 'deg90', 'deg135']
    #glrlm: output,the glrlm result

    P = array
    x, y = P.shape
    min_pixels = np.min(P)   # the min pixel
    run_length = max(x, y)   # Maximum parade length in pixels
    num_level = np.max(P) - np.min(P) + 1   # Image gray level

    deg0 = [val.tolist() for sublist in np.vsplit(P, x) for val in sublist]   # 0deg
    deg90 = [val.tolist() for sublist in np.split(np.transpose(P), y) for val in sublist]   # 90deg
    diags = [P[::-1, :].diagonal(i) for i in range(-P.shape[0]+1, P.shape[1])]   #45deg
    deg45 = [n.tolist() for n in diags]
    Pt = np.rot90(P, 3)   # 135deg
    diags = [Pt[::-1, :].diagonal(i) for i in range(-Pt.shape[0]+1, Pt.shape[1])]
    deg135 = [n.tolist() for n in diags]

    def length(l):
        if hasattr(l, '_len_'):
            return np.size(l)
        else:
            i = 0
            for _ in l:
                i += 1
            return i

    glrlm = np.zeros((num_level, run_length, len(theta)))
    for angle in theta:
        for splitvec in range(0, len(eval(angle))):
            flattened = eval(angle)[splitvec]
            answer = []
            for key, iter in groupby(flattened):
                answer.append((key, length(iter)))
            for ansIndex in range(0, len(answer)):
                glrlm[int(answer[ansIndex][0]-min_pixels), \
                      int(answer[ansIndex][1]-1), theta.index(angle)] += 1
    return glrlm

def generate_2ndOrder_RunLength_matrix_features(ROI):
    theta = [['deg0'], ['deg45'], ['deg90'], ['deg135']]
    SRE_l = []
    LRE_l = []
    GLN_l = []
    RLN_l = []
    RP_l = []
    RL_feats = []
    for deg in theta:
        now_deg = deg[0]
        test_data = generate_RL_matrix(ROI, deg)



        # 2
        LRE = getLongRunEmphasis(test_data)
        LRE = np.squeeze(LRE)
        LRE_l.append(LRE)


        RL_feats.append([LRE])
        # print(now_deg,RL_feats)
        # print("=============")

    RL_feats = np.asarray(RL_feats)

    # RL_matrix=getGrayLevelRumatrix(ROI,theta[0])
    # RL_matrix=np.asarray(RL_matrix)

    # print(RL_matrix)
    # print(RL_feats)


    LRE_vals = RL_feats[:, 0];


    LRE_mean = np.mean(LRE_vals);
    LRE_range = np.max(LRE_vals) - np.min(LRE_vals)

    features = np.zeros(1)
    feature_names = ['LRE_range'
                     ]
    feature_names = np.asarray(feature_names)

    features[0] = LRE_range

    # print(features)

    return (feature_names, features)

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


def generate_2ndOrder_coocurrence_matrix_features(im):
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
    feature_names = [
                     'correlation_mean'
                     ]

    correlation_mean = np.mean(correlation);


    features[0] = correlation_mean;

    # print(features)
    return (feature_names, features)


def generate_first_order_features_py(m):
    from scipy import stats
    N=np.size(m,0);M=np.size(m,1)
    features=np.zeros(1)
    features[0]=np.mean(m)

    feature_names=['mean']#define feature names to columns
    feature_names=np.asarray(feature_names)
    return(feature_names,features)