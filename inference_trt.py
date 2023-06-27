import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import numpy as np
from numpy import random
from exec_backends.trt_loader import TrtModelNMS
import torch
# from utils import *
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import ops
# from models.models import Darknet


retina_masks = True
conf = 0.25
iou = 0.7
agnostic_nms = False

def postprocess(preds, img, orig_imgs, retina_masks, conf, iou, agnostic_nms=False):
    """TODO: filter by classes."""
    
    p = ops.non_max_suppression(preds[0],
                                conf,
                                iou,
                                agnostic_nms,
                                max_det=300,
                                nc=1)



    results = []
    proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # second output is len 3 if pt, but only 1 if exported
    for i, pred in enumerate(p):
        orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
        # path = self.batch[0]
        img_path = "ok"
        if not len(pred):  # save empty boxes
            results.append(Results(orig_img=orig_img, path=img_path, names="segment", boxes=pred[:, :6]))
            continue
        if retina_masks:
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
        else:
            masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        results.append(
            Results(orig_img=orig_img, path=img_path, names="1213", boxes=pred[:, :6], masks=masks))
    return results

class FastSam(object):
    def __init__(self, 
            model_weights = '/models/Yolo-nas/weights/coco_yolonas.trt', 
            max_size = 640):
        # self.names = [f"tattoo{i}" for i in range(80)]
        # self.names = load_classes(names)
        # self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        self.imgsz = (max_size, max_size)
        # Load model
        self.model = TrtModelNMS(model_weights, max_size)


    def segment(self, bgr_img):
        # input, (x_ratio, y_ratio) =  preprocess(bgr_img, (416, 416))
        # print(input.shape)   
        # Prediction
        ## Padded resize
        h, w, _ = bgr_img.shape
        # bgr_img, _, _ = letterbox(bgr_img)
        scale = min(self.imgsz[0]/w, self.imgsz[1]/h)
        inp = np.zeros((self.imgsz[1], self.imgsz[0], 3), dtype = np.float32)
        nh = int(scale * h)
        nw = int(scale * w)
        inp[: nh, :nw, :] = cv2.resize(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB), (nw, nh))
        inp = inp.astype('float32')# / 255.0  # 0 - 255 to 0.0 - 1.0
        inp = np.expand_dims(inp.transpose(2, 0, 1), 0)
        # print(inp.shape)  
        # print(x_ratio, y_ratio)
        ## Inference
        t1 = time.time()
        print("[Input]: ", inp[0].transpose(0, 1, 2).shape)
        # cv2.imwrite("test.jpg", inp[0].transpose(1, 2, 0) * 255)
        # mask, proto, _, _, _, _ = self.model.run(inp)
        preds = self.model.run(inp)
        print([x.shape for x in preds])
        exit(1)
        data_0 = torch.from_numpy(preds[5])
        data_1 = [[torch.from_numpy(preds[2]), torch.from_numpy(preds[3]), torch.from_numpy(preds[4])], torch.from_numpy(preds[1]), torch.from_numpy(preds[0])]
        preds = [data_0, data_1]
        result = postprocess(preds, inp, bgr_img)
        print(result)
        exit(1)
        
        # print(masks)
        # exit(1)
        # proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # second output is len 3 if pt, but only 1 if exported
        
        # # proto = point_1
        # for i, pred in enumerate(preds[0]):
        #     if not isinstance(bgr_img, torch.Tensor):
        #         pred[:, :4] = scale_boxes(inp.shape[2:], pred[:, :4], bgr_img.shape)
        #         print(pred[:, :4].shape)
        #     masks = process_mask_native(proto[i], pred[:, 6:], pred[:, :4], bgr_img.shape[:2])  # HWC
        #     print(masks)
        #     print(masks.shape)
        # print(point_1.shape)

        # t2 = time.time()
        # print('Time cost: ', t2 - t1)
        # ## Apply NMS
        # num_detection = num_detection[0][0]
        # nmsed_bboxes  = nmsed_bboxes[0]
        # nmsed_scores  = nmsed_scores[0]
        # nmsed_classes  = nmsed_classes[0]
        # print(num_detection)
        # # print(nmsed_classes)
        # print('Detected {} object(s)'.format(num_detection))
        # # print(nmsed_bboxes[:2])
        # for bbx in nmsed_bboxes[:2]:
        #     print(bbx)
        # # Rescale boxes from img_size to im0 size
        # _, _, height, width = inp.shape
        # h, w, _ = bgr_img.shape
        # nmsed_bboxes[:, 0] /= scale
        # nmsed_bboxes[:, 1] /= scale
        # nmsed_bboxes[:, 2] /= scale
        # nmsed_bboxes[:, 3] /= scale
        # visualize_img = bgr_img.copy()
        # for ix in range(num_detection):       # x1, y1, x2, y2 in pixel format
        #     cls = int(nmsed_classes[ix])
        #     label = '%s %.2f' % (self.names[cls], nmsed_scores[ix])
        #     x1, y1, x2, y2 = nmsed_bboxes[ix]

        #     cv2.rectangle(visualize_img, (int(x1), int(y1)), (int(x2), int(y2)), self.colors[int(cls)], 2)
        #     cv2.putText(visualize_img, label, (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors[int(cls)], 2, cv2.LINE_AA)

        # cv2.imwrite('result.jpg', visualize_img)
        return mask, point_1

if __name__ == '__main__':
    model = FastSam(model_weights="/models/FastSam/fast_sam.trt")
    img = cv2.imread('/models/FastSam/cat.jpg')
    mask, point_1 = model.segment(img)
    
    # print(mask)
    print("[Ouput 0]: ", mask.shape)
    # print(point_1)
    print("[Ouput 1]: ", point_1.shape)

