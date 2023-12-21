import cv2
import numpy as np
from time import time

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
    equalLabels_list = {}
    # 4-Neighbors
    directs = [[0,-1],[0,+1],[1,0],[-1,0]]

    # One Pass
    for col in range(cols):
        for row in range(rows):
            if img[col, row] == 0: continue
            currLabel_list = [[col,row]]
            isNewLabel = True
            min_label = label
            neighbors = []
            # iterate 4 neighbors (actually, 2 neighbors are only needed)
            for [offX, offY] in directs:
                if col+offX < 0 or col+offX >=cols or row+offY <0 or row+offY>= rows: continue

                neig_label = label_mask[col+ offX, row+ offY] 
                if neig_label > 0:
                    
                    # add neighbor to Set
                    currLabel_list += equalLabels_list[neig_label]
                    neighbors += [neig_label]

                    isNewLabel = False
                    # assign minimun label
                    if min_label > neig_label:
                        min_label = neig_label

            label_mask[col, row] = min_label

            if isNewLabel:
                label+=1
            
            equalLabels_list[min_label] = currLabel_list
            
            for ellist in neighbors:
                if ellist == min_label: continue
                equalLabels_list.pop(ellist)
                
    

    # Two Pass
    for (k, var) in equalLabels_list.items():
        
        for [col, row] in var:
            label_mask[col,row] =  k
    
    return label_mask

# K-means
# input is a Binary Img
def Kmeans_XY(img, k=5):
    
    cols, rows = np.shape(img)
    centers_x = np.random.randint(0, cols, size=(k,1))
    centers_y = np.random.randint(0, rows, size=(k,1))
    centers = np.column_stack((centers_x, centers_y))
    
    label_mask = np.zeros((cols,rows), np.uint8)
    isConverged = False
    count = 0
    
    del_mean = np.mean(centers)
    while not isConverged:
        start_t = time()
        # iterate each valued points and classify it to a label
        for col in range(cols):
            for row in range(rows):
                if img[col, row] == 0: continue

                min_idx = 0
                min_dist = np.sqrt((cols)**2+(rows)**2)
                # for current pixel, cal the distance to each centers, and find the closest center
                for cls_id, [cx, cy] in enumerate(centers):
                    dist = np.sqrt((cx-col)**2+(cy-row)**2)
                    if min_dist > dist:
                        min_dist = dist
                        min_idx = cls_id+1

                # assign class label to current pixel        
                label_mask[col, row] = min_idx

        # re-calculate the centers based on the labels
        for cls_id in range(k):
            cls_id +=1
            c_acc, r_acc = 0,0
            locations = np.argwhere(label_mask == cls_id)
            for [col, row] in locations:
                c_acc+= col
                r_acc+= row
            if len(locations) ==0:
                c_mean =0
                r_mean =0
            else:
                c_mean = c_acc/ len(locations)
                r_mean = r_acc/ len(locations)

            centers[cls_id-1,:] = [c_mean, r_mean]
        
        count+=1
        end_t = time()
        curr_mean = np.mean(centers)
        print("processed %d times, elapsed time: %.3f" % (count, end_t - start_t))
        del_mean = np.abs(np.abs(del_mean) - np.abs(curr_mean))
        print("delta mean %.3f" % (del_mean))
        # if del_mean < 1.0 : isConverged=True

        del_mean = curr_mean


    return label_mask

# input is a Gray Img
def Kmeans_inten(img, k=5):

    cols, rows = np.shape(img)
    centers = np.random.rand(k,1)*255


    isConverged = False
    count = 0
    label_mask = np.zeros((cols,rows), np.uint8)
    del_mean = np.mean(centers)
    while not isConverged:
        start_t = time()
        
        for row in range(rows):
            for col in range(cols):
                curr_pixel = img[col, row]

                min_dist = 256
                min_labelId = 1
                for labelId, c_inten in enumerate(centers):
                    dist = np.sqrt((curr_pixel-c_inten[0])**2)
                    if dist < min_dist:
                        min_labelId = labelId+1
                        min_dist = dist
                
                label_mask[col, row] = min_labelId
        
        for labelId in range(1, len(centers)+1):
            locations = np.argwhere(label_mask == labelId)
            inten_acc = 0.0
            for [col, row] in locations:
                    inten_acc += img[col, row]

            if len(locations) ==0:
                inten_acc =0
            else:
                inten_acc /= len(locations)
            centers[labelId-1] = inten_acc
        
        count+=1
        end_t = time()
        curr_mean = np.mean(centers)
        print("processed %d times, elapsed time: %.3f" % (count, end_t - start_t))
        del_mean = np.abs(np.abs(del_mean) - np.abs(curr_mean))
        print("delta mean %.4f" % (del_mean))
        if del_mean < 0.01 : isConverged=True

        del_mean = curr_mean

    return label_mask              








