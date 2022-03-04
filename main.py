import cv2
import numpy as np
import math
from cuml.cluster import KMeans as KMeans
import matplotlib.pyplot as plt

def compress_img(img, k):
    
    # Step 1: reshape
    h, w, _ = img.shape
    vectors = img.reshape(h*w, 3)
    vectors = np.asarray(vectors, dtype=np.float32)
    
    # Step 2: find k cluster
    km = KMeans(n_clusters=k)
    km.fit(vectors)
    
    # Step 3: replace every data point to the nearest cluster with itself
    compressed_vectors = km.cluster_centers_[km.labels_]
    compressed_vectors = np.clip(compressed_vectors.astype('uint8'), 0, 255)
    compressed_img = compressed_vectors.reshape(h, w, 3)
    
    # cal compress ratio
    r = (24*k + h * w * math.ceil(math.log(k, 2))) / (24 * h * w)
    
    return r, compressed_img


if __name__ == '__main__':
    
    # modify image path at here
    #img_path = '/images/'
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    
    r, compressed_img = compress_img(img, k=64)
    print('compress_ratio = ', r)
    cv2.imwrite('/images/compressed_image.png', compressed_img)
