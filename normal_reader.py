# for debugging.
import cv2
import numpy as np
from tqdm import tqdm
p = np.load("results/generated_with_normal_1/0/normal.npy")
p = np.abs(p)
mask = (np.max(p, axis=2) > 0).astype('int')
mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
mask = np.repeat(mask, 3, axis=2)
debug = open("debug.txt","w")
for i in tqdm(range(mask.shape[0])):
    for j in range(mask.shape[1]):
        debug.write(str(p[i, j, 0])+" ")
    debug.write("\n")
    debug.flush()
debug.close()
color = mask * (p + 1) * 0.5
# color = np.floor(mask * (255 * ((p + 1) * 0.5) + 0.5))
cv2.imshow("color", color)
cv2.waitKey()
cv2.imwrite("results/generated_with_normal_1/0/normal.png", color)
print(p)
