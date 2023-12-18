import cv2
import numpy as np


# Otsu thresholding, auto thresholding img
def OtsuThreshold(img):

    gray_level = 256

    # cal mean of  each gray level
    numOfpixels = np.size(img)
    gray_p = np.zeros(gray_level)

    for l in range(gray_level):
        acc = np.sum(img==l)
        gray_p[l] = acc
    
    gray_p /= numOfpixels

    max_g =0.0
    th_otsu = 0
    for t in range(gray_level):

        w1 = np.sum(gray_p[0:t])
        w2 = 1 - w1
        if w1 ==0 or w2 ==0: continue

        u1 = 0.0
        u2 = 0.0
        for fp in range(t):
            u1 += (fp * gray_p[fp])
        u1 = u1/w1
        for bp in range(t, gray_level):
            u2 += (bp * gray_p[bp])
        u2 = u2/w2

        # 
        G = w1*w2*(u1-u2)**2

        if max_g < G:
            max_g = G
            th_otsu = t
    img[img >= th_otsu] = 255
    img[img < th_otsu] = 0

    return img

# Two Pass connected component
# input is a Binary Img
def TwoPassCComp(img):

    cols, rows = np.shape(img)

    label_mask = np.zeros(np.shape(img), np.uint16)

    label = 1
    equalLabels_list = []
    # 4-Neighbors
    directs = [[0,-1],[0,+1],[1,0],[-1,0]]

    # One Pass
    for col in range(cols):
        for row in range(rows):
            if img[col, row] == 0: continue
            currLabel_list = []
            isNewLabel = True
            min_label = label

            # iterate 4 neighbors (actually, 2 neighbors are only needed)
            for [offX, offY] in directs:
                if col+offX < 0 or col+offX >=cols or row+offY <0 or row+offY>= rows: continue

                neig_label = label_mask[col+ offX, row+ offY] 
                if neig_label > 0:
                    
                    # add neighbor to Set
                    currLabel_list.append(neig_label)

                    isNewLabel = False
                    # assign minimun label
                    if min_label > neig_label:
                        min_label = neig_label

            label_mask[col, row] = min_label

            if isNewLabel:
                label+=1
                currLabel_list.append(min_label)
            
            equalLabels_list.append(currLabel_list)
                
    # remove duplicate pairs
    list_result =[]
    for baselist in equalLabels_list:
        b_set = set(baselist)
        if len(b_set) ==0: continue
        isfound = True
        while isfound:
            isfound = False
            for idx in range(len(equalLabels_list)):
                tar_set = set(equalLabels_list[idx])
                if len(tar_set) ==0: continue
                if b_set.isdisjoint(tar_set) == False:
                    b_set.update(tar_set)
                    equalLabels_list[idx] = []
                    isfound = True
                    break

                 
        list_result.append(list(b_set))         

    # Two Pass
    for col in range(cols):
        for row in range(rows):
            if label_mask[col, row] == 0: continue

            for equalLabel in list_result:
                if  label_mask[col, row] in equalLabel:
                    label_mask[col, row] = min(equalLabel)
    
    return label_mask

            
            


            

        
                
                    








