# encoding:utf-8
import os
import glob
import time
import json
import csv
import traceback
import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils.utils_tool import logger
from utils.data_provider.data_util import GeneratorEnqueuer
import tensorflow as tf
import pyclipper

tf.app.flags.DEFINE_string('training_data_path', None,
                           'training dataset to use')
tf.app.flags.DEFINE_integer('max_image_large_side', 1280,
                            'max image size of training')
tf.app.flags.DEFINE_integer('max_text_size', 800,
                            'if the text in the input image is bigger than this, then we resize'
                            'the image according to this')
tf.app.flags.DEFINE_integer('min_text_area_size', 10,
                            'if the text area size is smaller than this, we ignore it during training')
tf.app.flags.DEFINE_float('min_crop_side_ratio', 0.1,
                          'when doing random crop from input image, the'
                          'min length of min(H, W')

FLAGS = tf.app.flags.FLAGS


def get_files(exts):
    files = []
    for ext in exts:
        files.extend(glob.glob(
            os.path.join(FLAGS.training_data_path, '*.{}'.format(ext))))
    return files

def get_json_label():
    label_file_list = get_files(['json'])
    label = {}
    for label_file in label_file_list:
        with open(label_file, 'r') as f:
            json_label = json.load(f)

            for k, v in json_label.items():
                if not label.has_key(k):
                    label[k] = v
    return label

def load_annoataion(p):
    '''
    load annotation from the text file
    :param p:
    :return:
    '''
    text_polys = []
    text_tags = []
    if not os.path.exists(p):
        return np.array(text_polys, dtype=np.float32)
    with open(p, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            label = line[-1]
            # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]

            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
            text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            #TODO:maybe add '?' for icpr2018 (michael)
            if label == '*' or label == '###' or label == '?':
                text_tags.append(True)
            else:
                text_tags.append(False)
        return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool)

def check_and_validate_polys(polys, tags, xxx_todo_changeme):
    '''
    check so that the text poly is in the same direction,
    and also filter some invalid polygons
    :param polys:
    :param tags:
    :return:
    '''
    (h, w) = xxx_todo_changeme
    if polys.shape[0] == 0:
        return [], []
    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w-1)
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h-1)

    validated_polys = []
    validated_tags = []
    for poly, tag in zip(polys, tags):
        if abs(pyclipper.Area(poly))<1:
            continue
        #clockwise
        if pyclipper.Orientation(poly):
            poly = poly[::-1]

        validated_polys.append(poly)
        validated_tags.append(tag)
    return np.array(validated_polys), np.array(validated_tags)

def crop_area(im, polys, tags, crop_background=False, max_tries=50):
    '''
    make random crop from the input image
    :param im:
    :param polys:
    :param tags:
    :param crop_background:
    :param max_tries:
    :return:
    '''
    h, w, _ = im.shape
    pad_h = h//10
    pad_w = w//10
    h_array = np.zeros((h + pad_h*2), dtype=np.int32)
    w_array = np.zeros((w + pad_w*2), dtype=np.int32)
    for poly in polys:
        poly = np.round(poly, decimals=0).astype(np.int32)
        minx = np.min(poly[:, 0])
        maxx = np.max(poly[:, 0])
        w_array[minx+pad_w:maxx+pad_w] = 1
        miny = np.min(poly[:, 1])
        maxy = np.max(poly[:, 1])
        h_array[miny+pad_h:maxy+pad_h] = 1
    # ensure the cropped area not across a text
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]
    if len(h_axis) == 0 or len(w_axis) == 0:
        return im, polys, tags
    for i in range(max_tries):
        xx = np.random.choice(w_axis, size=2)
        xmin = np.min(xx) - pad_w
        xmax = np.max(xx) - pad_w
        xmin = np.clip(xmin, 0, w-1)
        xmax = np.clip(xmax, 0, w-1)
        yy = np.random.choice(h_axis, size=2)
        ymin = np.min(yy) - pad_h
        ymax = np.max(yy) - pad_h
        ymin = np.clip(ymin, 0, h-1)
        ymax = np.clip(ymax, 0, h-1)
        if xmax - xmin < FLAGS.min_crop_side_ratio*w or ymax - ymin < FLAGS.min_crop_side_ratio*h:
            # area too small
            continue
        if polys.shape[0] != 0:
            poly_axis_in_area = (polys[:, :, 0] >= xmin) & (polys[:, :, 0] <= xmax) \
                                & (polys[:, :, 1] >= ymin) & (polys[:, :, 1] <= ymax)
            selected_polys = np.where(np.sum(poly_axis_in_area, axis=1) == 4)[0]
        else:
            selected_polys = []
        if len(selected_polys) == 0:
            # no text in this area
            if crop_background:
                return im[ymin:ymax+1, xmin:xmax+1, :], polys[selected_polys], tags[selected_polys]
            else:
                continue
        im = im[ymin:ymax+1, xmin:xmax+1, :]
        polys = polys[selected_polys]
        tags = tags[selected_polys]
        polys[:, :, 0] -= xmin
        polys[:, :, 1] -= ymin
        return im, polys, tags

    return im, polys, tags

def perimeter(poly):
    try:
        p=0
        nums = poly.shape[0]
        for i in range(nums):
            p += abs(np.linalg.norm(poly[i%nums]-poly[(i+1)%nums]))
        # logger.debug('perimeter:{}'.format(p))
        return p
    except Exception as e:
        traceback.print_exc()
        raise e

def shrink_poly(poly, r):
    try:
        area_poly = abs(pyclipper.Area(poly))
        perimeter_poly = perimeter(poly)
        poly_s = []
        pco = pyclipper.PyclipperOffset()
        if perimeter_poly:
            d=area_poly*(1-r*r)/perimeter_poly
            pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            poly_s = pco.Execute(-d)
        return poly_s
    except Exception as e:
        traceback.print_exc()
        raise e

#TODO:filter small text(when shrincked region shape is 0 no matter what scale ratio is)
def generate_seg(im_size, polys, tags, image_name, scale_ratio):
    '''
    :param im_size: input image size
    :param polys: input text regions
    :param tags: ignore text regions tags
    :param image_index: for log
    :param scale_ratio:ground truth scale ratio, default[0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    :return:
    seg_maps: segmentation results with different scale ratio, save in different channel
    training_mask: ignore text regions
    '''
    h, w = im_size
    #mark different text poly
    seg_maps = np.zeros((h,w,6), dtype=np.uint8)
    # mask used during traning, to ignore some hard areas
    training_mask = np.ones((h, w), dtype=np.uint8)
    ignore_poly_mark = []
    for i in range(len(scale_ratio)):
        seg_map = np.zeros((h,w), dtype=np.uint8)
        for poly_idx, poly_tag in enumerate(zip(polys, tags)):
            poly = poly_tag[0]
            tag = poly_tag[1]

            # ignore ###
            if i == 0 and tag:
                cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
                ignore_poly_mark.append(poly_idx)

            # seg map
            shrinked_polys = []
            if poly_idx not in ignore_poly_mark:
                shrinked_polys = shrink_poly(poly.copy(), scale_ratio[i])

            if not len(shrinked_polys) and poly_idx not in ignore_poly_mark:
                logger.info("before shrink poly area:{} len(shrinked_poly) is 0,image {}".format(
                    abs(pyclipper.Area(poly)),image_name))
                # if the poly is too small, then ignore it during training
                cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
                ignore_poly_mark.append(poly_idx)
                continue
            for shrinked_poly in shrinked_polys:
                seg_map = cv2.fillPoly(seg_map, [np.array(shrinked_poly).astype(np.int32)], 1)

        seg_maps[..., i] = seg_map
    return seg_maps, training_mask


def generator(input_size=512, batch_size=32,
              background_ratio=3./8,
              random_scale=np.array([0.125, 0.25,0.5, 1, 2.0, 3.0]),
              vis=False,
              scale_ratio=np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])):
    '''
    reference from https://github.com/argman/EAST
    :param input_size:
    :param batch_size:
    :param background_ratio:
    :param random_scale:
    :param vis:
    :param scale_ratio:ground truth scale ratio
    :return:
    '''
    image_list = np.array(get_files(['jpg', 'png', 'jpeg', 'JPG']))

    logger.info('{} training images in {}'.format(
        image_list.shape[0], FLAGS.training_data_path))
    index = np.arange(0, image_list.shape[0])

    while True:
        np.random.shuffle(index)
        images = []
        image_fns = []
        seg_maps = []
        training_masks = []
        for i in index:
            try:
                im_fn = image_list[i]
                im = cv2.imread(im_fn)
                if im is None:
                    logger.info(im_fn)
                h, w, _ = im.shape
                txt_fn = im_fn.replace(os.path.basename(im_fn).split('.')[1], 'txt')
                if not os.path.exists(txt_fn):
                    continue

                text_polys, text_tags = load_annoataion(txt_fn)
                if text_polys.shape[0] == 0:
                    continue
                text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (h, w))

                # random scale this image
                rd_scale = np.random.choice(random_scale)
                im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
                text_polys *= rd_scale
                # random crop a area from image
                if np.random.rand() < background_ratio:
                    # crop background
                    im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=True)
                    if text_polys.shape[0] > 0:
                        # cannot find background
                        continue
                    # pad and resize image
                    new_h, new_w, _ = im.shape
                    #max_h_w_i = np.max([new_h, new_w, input_size])
                    im_padded = np.zeros((new_h, new_w, 3), dtype=np.uint8)
                    im_padded[:new_h, :new_w, :] = im.copy()
                    im = cv2.resize(im_padded, dsize=(input_size, input_size))
                    seg_map_per_image = np.zeros((input_size, input_size, scale_ratio.shape[0]), dtype=np.uint8)
                    training_mask = np.ones((input_size, input_size), dtype=np.uint8)
                else:
                    im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=False)
                    if text_polys.shape[0] == 0:
                        continue
                    # h, w, _ = im.shape

                    # pad the image to the training input size or the longer side of image
                    new_h, new_w, _ = im.shape
                    #max_h_w_i = np.max([new_h, new_w, input_size])
                    im_padded = np.zeros((new_h, new_w, 3), dtype=np.uint8)
                    im_padded[:new_h, :new_w, :] = im.copy()
                    im = im_padded
                    # resize the image to input size
                    new_h, new_w, _ = im.shape
                    resize_h = input_size
                    resize_w = input_size
                    im = cv2.resize(im, dsize=(resize_w, resize_h))
                    resize_ratio_3_x = resize_w/float(new_w)
                    resize_ratio_3_y = resize_h/float(new_h)
                    text_polys[:, :, 0] *= resize_ratio_3_x
                    text_polys[:, :, 1] *= resize_ratio_3_y
                    new_h, new_w, _ = im.shape
                    seg_map_per_image, training_mask = generate_seg((new_h, new_w), text_polys, text_tags,
                                                                     image_list[i], scale_ratio)
                    if not len(seg_map_per_image):
                        logger.info("len(seg_map)==0 image: %d " % i)
                        continue

                if vis:
                    fig, axs = plt.subplots(3, 3, figsize=(20, 30))
                    axs[0, 0].imshow(im[..., ::-1])
                    axs[0, 0].set_xticks([])
                    axs[0, 0].set_yticks([])
                    axs[0, 1].imshow(seg_map_per_image[..., 0])
                    axs[0, 1].set_xticks([])
                    axs[0, 1].set_yticks([])
                    axs[0, 2].imshow(seg_map_per_image[..., 1])
                    axs[0, 2].set_xticks([])
                    axs[0, 2].set_yticks([])
                    axs[1, 0].imshow(seg_map_per_image[..., 2])
                    axs[1, 0].set_xticks([])
                    axs[1, 0].set_yticks([])
                    axs[1, 1].imshow(seg_map_per_image[..., 3])
                    axs[1, 1].set_xticks([])
                    axs[1, 1].set_yticks([])
                    axs[1, 2].imshow(seg_map_per_image[..., 4])
                    axs[1, 2].set_xticks([])
                    axs[1, 2].set_yticks([])
                    axs[2, 0].imshow(seg_map_per_image[..., 5])
                    axs[2, 0].set_xticks([])
                    axs[2, 0].set_yticks([])
                    axs[2, 1].imshow(training_mask)
                    axs[2, 1].set_xticks([])
                    axs[2, 1].set_yticks([])
                    plt.tight_layout()
                    plt.show()
                    plt.close()

                images.append(im[..., ::-1].astype(np.float32))
                image_fns.append(im_fn)
                seg_maps.append(seg_map_per_image[::4, ::4, :].astype(np.float32))
                training_masks.append(training_mask[::4, ::4, np.newaxis].astype(np.float32))

                if len(images) == batch_size:
                    yield images, image_fns, seg_maps,  training_masks
                    images = []
                    image_fns = []
                    seg_maps = []
                    training_masks = []
            except Exception as e:
                traceback.print_exc()
                continue


def get_batch(num_workers, **kwargs):
    try:
        enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=True)
        enqueuer.start(max_queue_size=24, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()


if __name__ == '__main__':
    gen = get_batch(num_workers=2, vis=True)
    while True:
        image, bbox, im_info = next(gen)
        logger.debug('done')
