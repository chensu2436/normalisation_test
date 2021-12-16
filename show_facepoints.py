import os
import cv2
# import numpy as np


with open("output.txt") as f:
    lines = f.readlines()

x = lines[1].split()
y = lines[2].split()

points = []
for i in range(49):
    points.append((int(float(x[i])),int(float(y[i]))))

print(points)

filepath = os.path.join('./data/example/day01_0087.jpg')
img_original = cv2.imread(filepath)

for point in points:
    cv2.drawMarker(img_original, point, (0,0,255), markerType=cv2.MARKER_STAR, 
    markerSize=20, thickness=1, line_type=cv2.LINE_AA)

cv2.imwrite('facepoints_49.jpg',img_original)

img_original = cv2.imread(filepath)
landmarks = [(551, 408), (603, 405), (698, 398), (755, 393), (603, 566), (724, 557)]
for point in landmarks:
    cv2.drawMarker(img_original, point, (0,0,255), markerType=cv2.MARKER_STAR, 
    markerSize=20, thickness=1, line_type=cv2.LINE_AA)

cv2.imwrite('facepoints_6.jpg',img_original)