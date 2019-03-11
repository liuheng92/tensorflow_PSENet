import logging
from easydict import EasyDict as edict
import Queue
import numpy as np
import cv2

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

__C = edict()
cfg = __C

#log level
__C.error = logging.ERROR
__C.warning = logging.WARNING
__C.info = logging.INFO
__C.debug = logging.DEBUG


def pse(kernals, min_area=5):
    '''
    reference https://github.com/whai362/PSENet/issues/15
    :param kernals:
    :param min_area:
    :return:
    '''
    kernal_num = len(kernals)
    if not kernal_num:
        logger.error('not kernals!')
        return np.array([]), []
    pred = np.zeros(kernals[0].shape, dtype='int32')

    label_num, label = cv2.connectedComponents(kernals[kernal_num - 1].astype(np.uint8), connectivity=4)
    label_values = []
    for label_idx in range(1, label_num):
        if np.sum(label == label_idx) < min_area:
            label[label == label_idx] = 0
            continue
        label_values.append(label_idx)

    queue = Queue.Queue(maxsize=0)
    next_queue = Queue.Queue(maxsize=0)
    points = np.array(np.where(label > 0)).transpose((1, 0))


    for point_idx in range(points.shape[0]):
        x, y = points[point_idx, 0], points[point_idx, 1]
        l = label[x, y]
        queue.put((x, y, l))
        pred[x, y] = l

    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    for kernal_idx in range(kernal_num - 2, -1, -1):
        kernal = kernals[kernal_idx].copy()
        while not queue.empty():
            (x, y, l) = queue.get()

            is_edge = True
            for j in range(4):
                tmpx = x + dx[j]
                tmpy = y + dy[j]
                if tmpx < 0 or tmpx >= kernal.shape[0] or tmpy < 0 or tmpy >= kernal.shape[1]:
                    continue
                if kernal[tmpx, tmpy] == 0 or pred[tmpx, tmpy] > 0:
                    continue

                queue.put((tmpx, tmpy, l))
                pred[tmpx, tmpy] = l
                is_edge = False
            if is_edge:
                next_queue.put((x, y, l))

        # kernal[pred > 0] = 0
        queue, next_queue = next_queue, queue

        # points = np.array(np.where(pred > 0)).transpose((1, 0))
        # for point_idx in range(points.shape[0]):
        #     x, y = points[point_idx, 0], points[point_idx, 1]
        #     l = pred[x, y]
        #     queue.put((x, y, l))

    return pred, label_values