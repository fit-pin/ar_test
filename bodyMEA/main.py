# 모션 트래킹 코드
from os import PathLike
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
from Yolo import resYolo

net = cv.dnn.readNetFromTensorflow("model/graph_opt.pb")

inWidth = 368
inHeight = 368
thr = 0.2

BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
              "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
              ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
              ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
              ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

img = cv.imread("bodyMEA/test.jpg")


def distance(point1: tuple, point2: tuple):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def pose(frame: cv.typing.MatLike):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    inp = cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight),
                               (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()

    assert (len(BODY_PARTS) <= out.shape[1])

    points = []
    arm = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponding body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]

        # Add a point if it's confidence is higher than threshold.

        points.append((int(x), int(y)) if conf > thr else None)
        count = 0
        if (BODY_PARTS["RShoulder"] == i):
            arm.append([x, y])
        if (BODY_PARTS["RElbow"] == i):
            arm.append([x, y])
        if (BODY_PARTS["RWrist"] == i):
            arm.append([x, y])

    print(arm)
    # for pair in POSE_PAIRS:
    #     partFrom = pair[0]
    #     partTo = pair[1]
    #     assert (partFrom in BODY_PARTS)
    #     assert (partTo in BODY_PARTS)

    #     idFrom = BODY_PARTS[partFrom]
    #     idTo = BODY_PARTS[partTo]

    #     cv.ellipse(frame, points[0], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
    #     cv.ellipse(frame, points[1], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
    #     print(distance(points[0], points[1]))

    #     if points[idFrom] and points[idTo]:
    #         cv.line(frame, points[idFrom], points[idTo], (255, 0, 0), 3)
    #         cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
    #         cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    return frame


est = pose(img)

img = resYolo(img)

cv.cvtColor(est, cv.COLOR_BGR2RGB)
cv.imwrite("bodyMEA/result.jpg", est)
plt.imshow(est)
plt.show()
