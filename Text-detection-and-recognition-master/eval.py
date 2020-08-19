# -*- coding:utf-8 -*-
import cv2
import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from utils.utils_tool import logger, cfg
import matplotlib.pyplot as plt
import sys
import pytesseract
# tf.get_logger().setLevel('INFO')
# tf.autograph.set_verbosity(3)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.app.flags.DEFINE_string('test_data_path', './images/', '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', '', '')
tf.app.flags.DEFINE_string('output_dir', './output/', '')
tf.app.flags.DEFINE_bool('no_write_images', False, 'do not write images')

from nets import model
from pse import pse

FLAGS = tf.app.flags.FLAGS

logger.setLevel(cfg.error)

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
    logger.info('Found {} images'.format(len(files)))
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


def detect(seg_maps, timer, image_w, image_h, min_area_thresh=10, seg_map_thresh=0.9, ratio = 1):
    '''
    restore text boxes from score map and geo map
    :param seg_maps:
    :param timer:
    :param min_area_thresh:
    :param seg_map_thresh: threshhold for seg map
    :param ratio: compute each seg map thresh
    :return:
    '''
    if len(seg_maps.shape) == 4:
        seg_maps = seg_maps[0, :, :, ]
    #get kernals, sequence: 0->n, max -> min
    kernals = []
    one = np.ones_like(seg_maps[..., 0], dtype=np.uint8)
    zero = np.zeros_like(seg_maps[..., 0], dtype=np.uint8)
    thresh = seg_map_thresh
    for i in range(seg_maps.shape[-1]-1, -1, -1):
        kernal = np.where(seg_maps[..., i]>thresh, one, zero)
        kernals.append(kernal)
        thresh = seg_map_thresh*ratio
    start = time.time()
    mask_res, label_values = pse(kernals, min_area_thresh)
    timer['pse'] = time.time()-start
    mask_res = np.array(mask_res)
    mask_res_resized = cv2.resize(mask_res, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
    boxes = []
    for label_value in label_values:
        #(y,x)
        points = np.argwhere(mask_res_resized==label_value)
        points = points[:, (1,0)]
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        boxes.append(box)

    return np.array(boxes), kernals, timer

def show_score_geo(color_im, kernels, im_res):
    fig = plt.figure()
    cmap = plt.cm.hot
    #
    ax = fig.add_subplot(241)
    im = kernels[0]*255
    ax.imshow(im)

    ax = fig.add_subplot(242)
    im = kernels[1]*255
    ax.imshow(im, cmap)

    ax = fig.add_subplot(243)
    im = kernels[2]*255
    ax.imshow(im, cmap)

    ax = fig.add_subplot(244)
    im = kernels[3]*255
    ax.imshow(im, cmap)

    ax = fig.add_subplot(245)
    im = kernels[4]*255
    ax.imshow(im, cmap)

    ax = fig.add_subplot(246)
    im = kernels[5]*255
    ax.imshow(im, cmap)

    ax = fig.add_subplot(247)
    im = color_im
    ax.imshow(im)

    ax = fig.add_subplot(248)
    im = im_res
    ax.imshow(im)

    fig.show()


def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list

    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise

    with tf.compat.v1.get_default_graph().as_default():
        input_images = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')# Create a tensorflow placeholder to hold image.
        global_step = tf.compat.v1.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        seg_maps_pred = model.model(input_images, is_training=False)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore()) 
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            # model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            # logger.info('Restore from {}'.format(model_path))
            # saver.restore(sess, model_path)
            saver = tf.train.Saver()
            saver.restore(sess,'./model/model.ckpt') # Using the already trained model restore its graph to saver.

            im_fn_list = get_images()# Call function get_images() to get the list of images in direcory.
            for im_fn in im_fn_list:
                im = cv2.imread(im_fn)[:, :, ::-1]# read the image in the folder.
                print("OG : ", cv2.imread(im_fn).shape)# print the Original dimension of the image.
                logger.debug('image file:{}'.format(im_fn))

                start_time = time.time()
                im_resized, (ratio_h, ratio_w) = resize_image(im) # resize the image to multiple of 32.
                im_resized = cv2.GaussianBlur(im_resized, (15,15),0)#Manually Add blur.
                cv2.imwrite(os.path.basename(im_fn)[:-4] + "_blur.jpg",im_resized) # Save the blurred image as a jpg file.
                h, w, _ = im_resized.shape
                # options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
                # run_metadata = tf.RunMetadata()
                timer = {'net': 0, 'pse': 0} # Used to measure the time model takes to detect.
                start = time.time()
                seg_maps = sess.run(seg_maps_pred, feed_dict={input_images: [im_resized]})
                timer['net'] = time.time() - start
                # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                # chrome_trace = fetched_timeline.generate_chrome_trace_format()
                # with open(os.path.join(FLAGS.output_dir, os.path.basename(im_fn).split('.')[0]+'.json'), 'w') as f:
                #     f.write(chrome_trace)

                boxes, kernels, timer = detect(seg_maps=seg_maps, timer=timer, image_w=w, image_h=h) # detect the text in the image and get the coordinates for the boxes.
                logger.info('{} : net {:.0f}ms, pse {:.0f}ms'.format(
                    im_fn, timer['net']*1000, timer['pse']*1000))

                if boxes is not None:
                    # print("BOXES 1 :",boxes)
                    boxes = boxes.reshape((-1, 4, 2))
                    # print("BOXES 2 :",boxes)
                    boxes[:, :, 0] /= ratio_w
                    boxes[:, :, 1] /= ratio_h
                    # print("BOXES 3 :",boxes)
                    h, w, _ = im.shape
                    boxes[:, :, 0] = np.clip(boxes[:, :, 0], 0, w)
                    boxes[:, :, 1] = np.clip(boxes[:, :, 1], 0, h)
                    # print("BOXES :",boxes)
                duration = time.time() - start_time
                logger.info('[timing] {}'.format(duration))# Print the total duration taken by the model.
                boxes_for_recognition = []
                results = []
                for box in boxes :
                    boxes_for_recognition.append([box[0][0],box[0][1],box[2][0],box[2][1]]) # Store the coordinates of recognised boxes in a list.
              # for i in range(len(boxes)):
                #     print("BOX : ",boxes[i])
                #     print("RECOG:",boxes_for_recognition[i])
                for box in boxes_for_recognition:
                    # Get the four coordinates of the bounding box where the text was detected.
                    startX = int(box[2] )
                    startY = int(box[3] )
                    endX = int(box[0])
                    endY = int(box[1])
                    if startX > endX :
                        startX,endX = endX,startX
                    if startY > endY :
                        startY,endY = endY,startY
                    
                    # in order to obtain a better OCR of the text we can potentially
                    # apply a bit of padding surrounding the bounding box -- here we
                    # are computing the deltas in both the x and y directions.

                    dX = int((endX - startX) * 0)#Here Padding is zero. 
                    dY = int((endY - startY) * 0)

                    # apply padding to each side of the bounding box, respectively
                    startX = max(0, startX - dX)
                    startY = max(0, startY - dY)
                    endX = min(w, endX + (dX * 2))
                    endY = min(h, endY + (dY * 2))
                    # print((startX, startY, endX, endY))
                    # extract the actual padded ROI
                    roi = im[startY:endY, startX:endX]

                    # in order to apply Tesseract v4 to OCR text we must supply
                    # (1) a language, (2) an OEM flag of 4, indicating that the we
                    # wish to use the LSTM neural net model for OCR, and finally
                    # (3) an OEM value, in this case, 7 which implies that we are
                    # treating the ROI as a single line of text
                    config = ("-l eng --oem 1 --psm 7")
                    text = pytesseract.image_to_string(roi, config=config)

                    # add the bounding box coordinates and OCR'd text to the list
                    # of results
                    results.append(((startX, startY, endX, endY), text))

                # sort the results bounding box coordinates from top to bottom
                results = sorted(results, key=lambda r:r[0][1])
                output = im.copy() 
                text_file = os.path.join(
                        "./recog_op/",
                        '{}.txt'.format(os.path.splitext(
                            os.path.basename(im_fn))[0]))# Open the text file in which the recognised text is stored.
                with open(text_file, 'w') as f:
                # loop over the results
                    for ((startX, startY, endX, endY), text) in results:
                        # display the text OCR'd by Tesseract
                        print("OCR TEXT")
                        print("========")
                        print("{}\n".format(text))
                        f.write("{}\n".format(text))
                        # strip out non-ASCII text so we can draw the text on the image
                        # using OpenCV, then draw the text and a bounding box surrounding
                        # the text region of the input image
                        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
                        
                        cv2.rectangle(output, (startX, startY), (endX, endY),
                            (0, 0, 255), 2)
                        # cv2.putText(output, text, (startX, startY - 20),
                        #     cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                        # show the output image
                        # cv2.imshow("Text Detection", output)
                        # img_path = os.path.join(FLAGS.output, os.path.basename(im_fn) + "op.jpg")
                cv2.imwrite("./recog_op/" + os.path.basename(im_fn)[:-4] + "op.jpg", output) #Save the output image with text boxes.
                    
                # save to file
                if boxes is not None:
                    res_file = os.path.join(
                        FLAGS.output_dir,
                        '{}.txt'.format(os.path.splitext(
                            os.path.basename(im_fn))[0]))


                    with open(res_file, 'w') as f:
                        num =0
                        for i in range(len(boxes)):
                            # to avoid submitting errors
                            box = boxes[i]
                            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                                continue

                            num += 1

                            f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                                box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1]))
                            cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=4)
                if not FLAGS.no_write_images:
                    img_path = os.path.join(FLAGS.output_dir, os.path.basename(im_fn))
                    cv2.imwrite(img_path, im[:, :, ::-1])
                # show_score_geo(im_resized, kernels, im)
if __name__ == '__main__':
    tf.compat.v1.app.run()
