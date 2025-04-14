import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans

# file_path = input('Enter filepath of the image:')
sample = r"C:\Users\aishw\OneDrive\Pictures\pexels-photo-733745.webp"
image_arr = mpimg.imread(sample)

plt.figure(dpi=200)
plt.imshow(image_arr)
(h,w,c) = image_arr.shape

image_2d = image_arr.reshape(h*w, c)
model = KMeans(n_clusters=24, random_state=100)
labels = model.fit_predict(image_2d)
rgb_codes = model.cluster_centers_.round(0).astype(int)
quantized_image = np.reshape(rgb_codes[labels], (h,w,c))
plt.imshow(quantized_image)
plt.savefig('quantized_out.jpeg')

