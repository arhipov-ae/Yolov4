# -*- coding: utf-8 -*-

import argparse
import os
import cv2
import numpy as np
from preprocessing import parse_annotation
from utils import get_annoboxes, draw_boxes
from frontend import YOLO
import json
from timeit import default_timer as timer
from PIL import Image
from my_scripts.convert_to_xml import save_anno_xml
import shutil

class predictor:

    def __init__(self, config_path, weights_path):
        with open(config_path) as config_buffer:
            config = json.loads(config_buffer.read())

        self.labels = config['model']['labels']

        self.yolo = YOLO(architecture     = config['model']['architecture'],
                        input_size        = config['model']['input_size'],
                        labels            = self.labels,
                        max_box_per_image = config['model']['max_box_per_image'],
                        anchors           = config['model']['anchors'])

        self.yolo.load_weights(weights_path)
        self.timing = [0, 0.]

    def _predict_one(self, image, threshold, decimals, draw_bboxes=True):

        t = timer()
        boxes = self.yolo.predict(image, threshold=threshold)
        image = draw_boxes(image, boxes, self.labels, decimals=decimals)
        t = timer() - t
        self.timing[0] += 1
        self.timing[1] += t
        print('{} boxes are found for {} s'.format(len(boxes), t))
        return image, boxes

    def predict_from_dir(self, path_to_dir, image_format, path_to_outputs = None, threshold=0.5, decimals=8, save_anno=False, draw_truth=False):
        if path_to_outputs and not os.path.exists(path_to_outputs):
            print('Creating output path {}'.format(path_to_outputs))
            os.mkdir(path_to_outputs)

        for image_filename in os.listdir(path_to_dir):
            # TODO: здесь надо сделать адекватную проверку, изображение ли это
            if image_filename.endswith(image_format):
                image = cv2.imread(os.path.join(path_to_dir, image_filename), cv2.IMREAD_COLOR)
                image_h = image.shape[0]
                image_w = image.shape[1]

                curr_time = timer()

                image, boxes = self._predict_one(image, threshold=threshold, decimals=decimals)


                curr_time = timer() - curr_time
                print(curr_time)

                boxes = get_annoboxes(image_w=image_w, image_h=image_h, boxes = boxes, labels=self.labels)

                if path_to_outputs:

                    if save_anno:
                        #
                        save_anno_xml(dir=path_to_outputs + 'annotations/',
                                      img_name=image_filename[:-len(image_format) - 1],
                                      img_format=image_format,
                                      img_w=image.shape[1],
                                      img_h=image.shape[0],
                                      img_d=image.shape[2],
                                      boxes=boxes,
                                      quiet=False,
                                      minConf=threshold)

                    retval = cv2.imwrite(path_to_outputs + 'images/' + image_filename, image)
                    if retval:
                        print('Изображение {} успешно сохранено в папку {}'.format(image_filename, path_to_outputs))
            else:
                print('В папке не только изображения - {}'.format(image_filename))

        print('Все изображения обработаны')
        print('Число изображений {}, общее время {}, среднее время на изображение {}'.format(self.timing[0], self.timing[1], self.timing[1]/self.timing[0]))


    def predict_from_webcam(self, threshold=0.5, fps=False, decimals=8):
        vid = cv2.VideoCapture(1)
        if not vid.isOpened():
            raise IOError(("Couldn't open webcam. If you're trying to open a webcam, "
                           "make sure you video_path is an integer!"))

        # Compute aspect ratio of video
        vidw = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        vidh = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        vidar = vidw / vidh


        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = timer()

        while True:
            retval, orig_image = vid.read()
            if not retval:
                print("Done!")
                return

            res_image = self._predict_one(orig_image, threshold=threshold, decimals=2)

            # Calculate FPS
            # This computes FPS for everything, not just the model's execution
            # which may or may not be what you want
            if fps:
                curr_time = timer()
                exec_time = curr_time - prev_time
                prev_time = curr_time
                accum_time = accum_time + exec_time
                curr_fps = curr_fps + 1
                if accum_time > 1:
                    accum_time = accum_time - 1
                    fps = "FPS: " + str(curr_fps)
                    curr_fps = 0

                # Draw FPS in top left corner
                cv2.rectangle(res_image, (0, 0), (50, 17), (255, 255, 255), -1)
                cv2.putText(res_image, fps, (3, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

            cv2.imshow("YOLOv2 result", res_image)
            pressedKey = cv2.waitKey(10)
            if pressedKey == 27:  # ESC key
                break

    def predict_from_video(self, path_to_video, threshold=0.5, decimals=8, output_file='', crop=True, writeFPS=False, show=False):
        vid = cv2.VideoCapture(path_to_video)
        if not vid.isOpened():
            raise IOError(("Couldn't open webcam. Make sure you video_path is an integer!"))

        # Compute aspect ratio of video
        vidw = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        vidh = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

        n = 0
        if crop:
            n = int((vidw - vidh) * 0.5)
            vidw = vidh

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_file, fourcc, 20.0, (int(vidw), int(vidh)))

        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = timer()

        while True:
            retval, orig_image = vid.read()
            if not retval:
                print("Done!")
                return

            if crop:
                orig_image = orig_image[:, n:int(n + vidh), :]

            res_image, boxes = self._predict_one(orig_image, threshold=threshold, decimals=decimals)

            # Calculate FPS
            # This computes FPS for everything, not just the model's execution
            # which may or may not be what you want
            if writeFPS:
                curr_time = timer()
                exec_time = curr_time - prev_time
                prev_time = curr_time
                accum_time = accum_time + exec_time
                curr_fps = curr_fps + 1
                if accum_time > 1:
                    accum_time = accum_time - 1
                    fps = "FPS: " + str(curr_fps)
                    curr_fps = 0

                # Draw FPS in top left corner
                cv2.rectangle(res_image, (0, 0), (50, 17), (255, 255, 255), -1)
                cv2.putText(res_image, fps, (3, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

            if show:
                cv2.imshow("YOLOv2 result", res_image)

            if output_file:
                out.write(res_image)

            pressedKey = cv2.waitKey(10)
            if pressedKey == 27:  # ESC key
                break

        out.release()



# A simple example how to use.
    # Create an object-predictor with certain parameters (in config file) and weights

'''
logdirs  = (20, 21, 22)
thres    = 0.5
videodir = '/media/data/ObjectDetectionExperiments/Datasets/9_TestNeuroMobile/orig/ЗНАКИ/'
outdir   = '/media/data/ObjectDetectionExperiments/Results/NeuroMobile/Signs/'
names    = os.listdir(videodir)

for logdir in logdirs:

    config = '../logs/{}/config2.json'.format(logdir)
    weights = '../logs/{}/weights.hdf5'.format(logdir)
    pred = predictor(config_path=config,
                     weights_path=weights)

    for name in names:
        outvideo = outdir + 'log{}_'.format(logdir) + name
        pred.predict_from_video(path_to_video=videodir+name,
                                threshold=thres,
                                output_file=outvideo,
                                decimals=2,
                                writeFPS=False)

'''
'''
logdir  = '15_signs_3'
config  =  '../logs/{}/config.json'.format(logdir)
weights = '../logs/{}/weights.hdf5'.format(logdir)
thress   = (0.25, 0.3, 0.35, 0.4)
l = len(thress)

pred = predictor(config_path=config,
                 weights_path=weights)

dir = '/home/user/Desktop/VideosNew/sign/'
out = '/home/user/Desktop/VideosNew/sign_res/'
flist = os.listdir(dir)
L = len(flist)

for i, f in enumerate(flist):
    videopath = dir + f

    for t, thres in enumerate(thress):
        resultpath = out + 'res_{}_{}.avi'.format(f[:f.rfind('.')], int(thres * 100))
        pred.predict_from_video(path_to_video=videopath,
                                threshold=thres,
                                output_file=resultpath,
                                crop=False,
                                decimals=2)
        print("{}/{}\t{}/{}".format(i+1, L, t+1, l))

'''
'''
def pred_docking(CONFIG_ID, DATASET_ID):
    main_path = "/media/ivan/Seagate Backup Plus Drive/check/docking-eval/docking_weights"
    configs = ['docking_full_yolo_finetune', 'docking_full_yolo_scratch', 'docking_mobilenet', 'docking_squeezenet_248',
               'docking_tiny_yolo_finetune',
               'docking_tiny_yolo_scratch_310']  # 'docking_tiny_yolo_scratch_50', 'docking_tiny_yolo_scratch_270',

    c_dir = configs[CONFIG_ID]

    config = '{0}/{1}/config.json'.format(main_path, c_dir)
    weights = '{0}/{1}/weights_last.hdf5'.format(main_path, c_dir)
    thress = (0.25, 0.3, 0.35, 0.4)
    l = len(thress)

    pred = predictor(config_path=config,
                     weights_path=weights)

    pred_main_path = "/media/ivan/Seagate Backup Plus Drive/check/docking-eval/imgs/"
    pred_dirs = ["node1", "node1-contr-rez", "node1-eqhist-contrast", "node1-contrast", "node1-eqhist", "node2",
                 "node3", "node4"]
    pred_dir = pred_dirs[DATASET_ID]
    result_main_path = "/media/ivan/Seagate Backup Plus Drive/check/docking-eval/yolo/"

    img_for_pred = '{0}/{1}/'.format(pred_main_path, pred_dir)
    img_results = '{0}/{1}__{2}/'.format(result_main_path, c_dir, pred_dir)

    if os.path.exists(img_results):
        shutil.rmtree(img_results, ignore_errors=True)
    os.makedirs(img_results)
    os.makedirs(img_results + 'annotations/')
    os.makedirs(img_results + 'images/')

    pred.predict_from_dir(path_to_dir=img_for_pred,
                          path_to_outputs=img_results,
                          image_format='bmp',
                          threshold=0.1,
                          save_anno=True,
                          decimals=2)


conf_ids = [0, 2]#0,1,2,3,4,5
dtst_ids = [5]#,5 0,1,2,3,4,6,7

for dtst_id in dtst_ids:
    for conf_id in conf_ids:
        pred_docking(conf_id, dtst_id)
################################
#CONFIG_ID=5  #[0-5]
#DATASET_ID=2 #[0-7]
################################
'''

config = '/media/ivan/Debian 7.8.0 i386 1/СВЕТА_ДИПЛОМ_ЗНАКИ/logs/21/config.json'
weights = '/media/ivan/Debian 7.8.0 i386 1/СВЕТА_ДИПЛОМ_ЗНАКИ/logs/21/weights.hdf5'

pred = predictor(config_path=config,
                     weights_path=weights)

img_for_pred = '/media/data/ObjectDetectionExperiments/Datasets/5_RTSD/ORIG/testimages/'
img_results  = '../logs/{}/out/'.format(logdir)

#img_for_pred = '/media/data/ObjectDetectionExperiments/Datasets/10_Helmet/images/val/'.format(dataset, d)
#img_results  = '/media/data/ObjectDetectionExperiments/Projects/2_YOLOs/YOLOv2_Orlova/Experiencor/Results/{}/'.format(logdir)

pred.predict_from_dir(path_to_dir=img_for_pred,
                      path_to_outputs=img_results,
                      image_format='jpg',
                      threshold=thres,
                      save_anno=False,
                      decimals=2)

print('Done!')

