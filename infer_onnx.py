import onnxruntime
import cv2
import numpy as np
import torch
from utils import overlay, segment_everything
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import ops
from PIL import Image
from random import randint

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

def preprocess(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (1024, 1024))
    rgb = np.array([rgb], dtype = np.float32) / 255.0
    return np.transpose(rgb, (0, 3, 1, 2))

if __name__ == '__main__':
    img = cv2.imread('/models/FastSam/cat.jpg')
    imgsz = [1024, 1024]
    # inp = preprocess(img)
    h, w, _ = img.shape
    # bgr_img, _, _ = letterbox(bgr_img)
    scale = min(imgsz[0]/w, imgsz[1]/h)
    inp = np.zeros((imgsz[1], imgsz[0], 3), dtype = np.float32)
    nh = int(scale * h)
    nw = int(scale * w)
    inp[: nh, :nw, :] = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (nw, nh))
    inp = inp.astype('float32')# / 255.0  # 0 - 255 to 0.0 - 1.0
    inp = np.expand_dims(inp.transpose(2, 0, 1), 0)
    # print(inp.shape)
    print('Input: ', inp.shape)
    model = onnxruntime.InferenceSession('/models/FastSam/fast_sam_1024.onnx', providers=['CUDAExecutionProvider'])
    ort_inputs = {model.get_inputs()[0].name: inp}
    preds = model.run(None, ort_inputs)
    print([x.shape for x in preds])
    data_0 = torch.from_numpy(preds[0])
    data_1 = [[torch.from_numpy(preds[1]), torch.from_numpy(preds[2]), torch.from_numpy(preds[3])], torch.from_numpy(preds[4]), torch.from_numpy(preds[5])]
    preds = [data_0, data_1]
    # for data in preds:
    #     if len(data)>0:
    #         print([i.shape for i in data])
    #     else:
    #         print(data.shape)
    result = postprocess(preds, inp, img, retina_masks, conf, iou)
    masks = result[0].masks.data
    
    print("len of mask: ", len(masks))
    # seg_value = 1
    image_with_masks = np.copy(img)
    # color_coverted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # # Displaying the converted image
    # pil_image = Image.fromarray(color_coverted)
    # # print()
    # image_with_masks = segment_everything(pil_image, result, input_size=1024)
    # image_with_masks.save("fast_sam_all.png")

    for i, mask_i in enumerate(masks):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (r, g, b)
        # cv2.rectangle(image_with_masks, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # cv2.putText(image_with_masks, str(round(float(scores[i]), 3)) ,(int(0.5*w), int(h-10)), 0, 2, (255,0,0), 1)
        image_with_masks = overlay(image_with_masks, mask_i, color=rand_color, alpha=1)
    cv2.imwrite("obj.png", image_with_masks)