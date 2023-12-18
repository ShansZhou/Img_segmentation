import cv2
import numpy as np
import img_segmentation as iseg


img_path = "./data/lena.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
cv2.imshow("Src", img)

# Otsu thresholding
src_img = np.copy(img)
img_thresh = iseg.OtsuThreshold(src_img)
cv2.imshow("Otsu", img_thresh)

# Two pass connected componet
src_img = np.copy(img_thresh)
src_img = cv2.resize(src_img, dsize=None, fx=0.1, fy=0.1)
# img_cc = iseg.TwoPassCComp(src_img)
tmp = np.array([[0,0,1,0,0,1,0],
          [1,1,1,0,1,1,1],
          [0,0,1,0,0,1,0],
          [0,1,1,0,1,1,0]])
img_cc = iseg.TwoPassCComp(tmp)
cv2.namedWindow("TwoPass", cv2.WINDOW_NORMAL)
cv2.imshow("TwoPass", img_cc)





cv2.waitKey(0)