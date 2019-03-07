# -*- coding:utf-8 -*-
import cv2
import time
import os
import numpy as np
import tensorflow as tf
from utils.utils_tool import logger, cfg
from PIL import Image
import matplotlib.pyplot as plt
import pylab

tf.app.flags.DEFINE_string('test_data_path', None, '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', './', '')
tf.app.flags.DEFINE_string('output_dir', './results/', '')
tf.app.flags.DEFINE_bool('no_write_images', False, 'do not write images')

from nets import model
from utils.utils_tool import pse

FLAGS = tf.app.flags.FLAGS

logger.setLevel(cfg.debug)

def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def resize_image(im, max_side_len=1200):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.

    #ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w


    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 + 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 + 1) * 32
    logger.info('resize_w:{}, resize_h:{}'.format(resize_w, resize_h))
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(seg_maps, timer, min_area_thresh=10, seg_map_thresh=0.8):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(seg_maps.shape) == 4:
        seg_maps = seg_maps[0, :, :, ]
    #get kernals, sequence: 0->n, max -> min
    kernals = []
    one = np.ones_like(seg_maps[..., 0], dtype=np.uint8)
    zero = np.zeros_like(seg_maps[..., 0], dtype=np.uint8)
    for i in range(seg_maps.shape[-1]-1, -1, -1):
        kernal = np.where(seg_maps[..., i]>seg_map_thresh, one, zero)
        kernals.append(kernal)
    start = time.time()
    mask_res = pse(kernals)
    timer['pse'] = time.time()-start
    _, contours0, hierarchy = cv2.findContours(mask_res.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours0:
        if cv2.contourArea(cnt) < min_area_thresh:
          continue
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        print box
        boxes.append(box)

    return np.array(boxes), timer

def show_score_geo(color_im, seg_maps, im_res):
    fig = plt.figure()
    cmap = plt.cm.hot
    seg_maps = seg_maps[0]*255
    #
    ax = fig.add_subplot(241)
    R = seg_maps[0, :, :, 0]*255
    im = (R)
    ax.imshow(im)

    ax = fig.add_subplot(242)
    R = seg_maps[0, :, :, 1]*255
    im = Image.fromarray(R)
    ax.imshow(im, cmap)

    ax = fig.add_subplot(243)
    R = seg_maps[0, :, :, 2]
    im = Image.fromarray(R)
    ax.imshow(im, cmap)

    ax = fig.add_subplot(244)
    R = seg_maps[0, :, :, 3]
    im = Image.fromarray(R)
    ax.imshow(im, cmap)

    ax = fig.add_subplot(245)
    R = seg_maps[0, :, :, 4]
    im = Image.fromarray(R)
    ax.imshow(im, cmap)

    ax = fig.add_subplot(246)
    R = seg_maps[0, :, :, 5]*255
    im = Image.fromarray(R)
    ax.imshow(im, cmap)

    ax = fig.add_subplot(247)
    im = Image.fromarray(color_im)
    ax.imshow(im)

    ax = fig.add_subplot(248)
    im = Image.fromarray( im_res )
    ax.imshow(im)

    pylab.show()


def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list

    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise

    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        seg_maps_pred = model.model(input_images, is_training=False)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            im_fn_list = get_images()
            for im_fn in im_fn_list:
                im = cv2.imread(im_fn)[:, :, ::-1]
                logger.debug('image file:{}'.format(im_fn))

                start_time = time.time()
                im_resized, (ratio_h, ratio_w) = resize_image(im)

                timer = {'net': 0, 'pse': 0}
                start = time.time()
                seg_maps = sess.run(seg_maps_pred, feed_dict={input_images: [im_resized]})
                timer['net'] = time.time() - start

                boxes, timer = detect(seg_maps=seg_maps, timer=timer)
                print('{} : net {:.0f}ms, pse {:.0f}ms'.format(
                    im_fn, timer['net']*1000, timer['pse']*1000))

                if boxes is not None:
                    print boxes.shape
                    boxes = boxes.reshape((-1, 4, 2))
                    boxes[:, :, 0] /= ratio_w
                    boxes[:, :, 1] /= ratio_h
                    h, w, _ = im.shape
                    boxes[:, :, 0] = np.clip(boxes[:, :, 0], 0, w)
                    boxes[:, :, 1] = np.clip(boxes[:, :, 1], 0, h)

                duration = time.time() - start_time
                print('[timing] {}'.format(duration))

                # save to file
                if boxes is not None:
                    res_file = os.path.join(
                        FLAGS.output_dir,
                        '{}.txt'.format(os.path.splitext(
                            os.path.basename(im_fn))[0]))


                    with open(res_file, 'w') as f:
                        num =0
                        for i in xrange(len(boxes)):
                            # to avoid submitting errors
                            box = boxes[i]
                            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                                continue

                            num += 1

                            f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                                box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1]))
                            cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=2)
                if not FLAGS.no_write_images:
                    img_path = os.path.join(FLAGS.output_dir, os.path.basename(im_fn))
                    cv2.imwrite(img_path, im[:, :, ::-1])
                # show_score_geo(im_resized, seg_maps, im)
if __name__ == '__main__':
    tf.app.run()
