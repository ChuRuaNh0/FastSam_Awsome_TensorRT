import time
import cv2
import numpy as np
from exec_backends.trt_loader import TrtModelNMS
import torch
from utils import overlay, segment_everything
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import ops
from random import randint
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
                                max_det=100,
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

def pre_processing(img_origin, imgsz=1024):
    h, w = img_origin.shape[:2]
    if h>w:
        scale   = min(imgsz / h, imgsz / w)
        inp     = np.zeros((imgsz, imgsz, 3), dtype = np.uint8)
        nw      = int(w * scale)
        nh      = int(h * scale)
        a = int((nh-nw)/2) 
        inp[: nh, a:a+nw, :] = cv2.resize(cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB), (nw, nh))
    else:
        scale   = min(imgsz / h, imgsz / w)
        inp     = np.zeros((imgsz, imgsz, 3), dtype = np.uint8)
        nw      = int(w * scale)
        nh      = int(h * scale)
        a = int((nw-nh)/2) 

        inp[a: a+nh, :nw, :] = cv2.resize(cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB), (nw, nh))
    rgb = np.array([inp], dtype = np.float32) / 255.0
    return np.transpose(rgb, (0, 3, 1, 2))

class FastSam(object):
    def __init__(self, 
            model_weights = '/models/fastSAm_wrapper/fast_sam_1024.trt', 
            max_size = 1024):
        self.imgsz = (max_size, max_size)
        # Load model
        self.model = TrtModelNMS(model_weights, max_size)


    def segment(self, bgr_img):
        ## Padded resize
        inp = pre_processing(bgr_img, self.imgsz[0])
        ## Inference
        t1 = time.time()
        print("[Input]: ", inp[0].transpose(0, 1, 2).shape)
        preds = self.model.run(inp)
        data_0 = torch.from_numpy(preds[5])
        data_1 = [[torch.from_numpy(preds[2]), torch.from_numpy(preds[3]), torch.from_numpy(preds[4])], torch.from_numpy(preds[1]), torch.from_numpy(preds[0])]
        preds = [data_0, data_1]

        result = postprocess(preds, inp, bgr_img)
        masks = result[0].masks.data
 
        image_with_masks = segment_everything(bgr_img, result, input_size=self.imgsz)

        cv2.imwrite(f"/models/FastSam/outputs/obj_segment_trt.png", image_with_masks)
        return masks

if __name__ == '__main__':
    model = FastSam(model_weights="/models/fastSAm_wrapper/fast_sam_1024.trt")
    img = cv2.imread('/models/FastSam/images/cat.jpg')
    masks = model.segment(img)
    print("[Ouput]: ", masks.shape)


