import torch.nn.functional as F
import torch
import torch
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2
import numpy as np

def clip_boxes(boxes, shape):
    """
    It takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the
    shape
    Args:
      boxes (torch.Tensor): the bounding boxes to clip
      shape (tuple): the shape of the image
    """
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    """
    Apply masks to bounding boxes using the output of the mask head.
    Args:
        protos (torch.Tensor): A tensor of shape [mask_dim, mask_h, mask_w].
        masks_in (torch.Tensor): A tensor of shape [n, mask_dim], where n is the number of masks after NMS.
        bboxes (torch.Tensor): A tensor of shape [n, 4], where n is the number of masks after NMS.
        shape (tuple): A tuple of integers representing the size of the input image in the format (h, w).
        upsample (bool): A flag to indicate whether to upsample the mask to the original image size. Default is False.
    Returns:
        (torch.Tensor): A binary mask tensor of shape [n, h, w], where n is the number of masks after NMS, and h and w
            are the height and width of the input image. The mask is applied to the bounding boxes.
    """

    c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)  # CHW

    downsampled_bboxes = bboxes.clone()
    downsampled_bboxes[:, 0] *= mw / iw
    downsampled_bboxes[:, 2] *= mw / iw
    downsampled_bboxes[:, 3] *= mh / ih
    downsampled_bboxes[:, 1] *= mh / ih

    masks = crop_mask(masks, downsampled_bboxes)  # CHW
    if upsample:
        masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]  # CHW
    return masks.gt_(0.5)


def process_mask_native(protos, masks_in, bboxes, shape):
    """
    It takes the output of the mask head, and crops it after upsampling to the bounding boxes.
    Args:
      protos (torch.Tensor): [mask_dim, mask_h, mask_w]
      masks_in (torch.Tensor): [n, mask_dim], n is number of masks after nms
      bboxes (torch.Tensor): [n, 4], n is number of masks after nms
      shape (tuple): the size of the input image (h,w)
    Returns:
      masks (torch.Tensor): The returned masks with dimensions [h, w, n]
    """
    c, mh, mw = protos.shape  # CHW
    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)
    gain = min(mh / shape[0], mw / shape[1])  # gain  = old / new
    pad = (mw - shape[1] * gain) / 2, (mh - shape[0] * gain) / 2  # wh padding
    top, left = int(pad[1]), int(pad[0])  # y, x
    bottom, right = int(mh - pad[1]), int(mw - pad[0])
    masks = masks[:, top:bottom, left:right]

    masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]  # CHW
    masks = crop_mask(masks, bboxes)  # CHW
    return masks.gt_(0.5)


def crop_mask(masks, boxes):
    """
    It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box
    Args:
      masks (torch.Tensor): [h, w, n] tensor of masks
      boxes (torch.Tensor): [n, 4] tensor of bbox coordinates in relative point form
    Returns:
      (torch.Tensor): The masks are being cropped to the bounding box.
    """
    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """
    Rescales bounding boxes (in the format of xyxy) from the shape of the image they were originally specified in
    (img1_shape) to the shape of a different image (img0_shape).
    Args:
      img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
      boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
      img0_shape (tuple): the shape of the target image, in the format of (height, width).
      ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
                         calculated based on the size difference between the two images.
    Returns:
      boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1), round(
            (img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

from typing import List, Tuple, Union
from torch import Tensor

def seg_postprocess(
        data: Tuple[Tensor],
        shape: Union[Tuple, List],
        conf_thres: float = 0.25,
        iou_thres: float = 0.65) \
        -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    assert len(data) == 2
    h, w = shape[0] // 4, shape[1] // 4  # 4x downsampling
    proto, outputs = (i[0] for i in data)
    # print(outputs.shape)
    # exit(1)
    bboxes, scores, labels, maskconf = outputs.split([4, 1, 1, 32], 1)
    scores, labels = scores.squeeze(), labels.squeeze()
    idx = scores > conf_thres
    if idx.sum() == 0:  # no bounding boxes or seg were created
        return None, None, None, None
    bboxes, scores, labels, maskconf = \
        bboxes[idx], scores[idx], labels[idx], maskconf[idx]
    idx = batched_nms(bboxes, scores, labels, iou_thres)
    bboxes, scores, labels, maskconf = \
        bboxes[idx], scores[idx], labels[idx].int(), maskconf[idx]
    masks = (maskconf @ proto).sigmoid().view(-1, h, w)
    masks = crop_mask(masks, bboxes / 4.)
    masks = F.interpolate(masks[None],
                          shape,
                          mode='bilinear',
                          align_corners=False)[0]
    masks = masks.gt_(0.5)[..., None]
    return bboxes, scores, labels, masks




class SegmentationPredictor(DetectionPredictor):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = 'segment'

    def postprocess(self, preds, img, orig_imgs):
        """TODO: filter by classes."""
        p = ops.non_max_suppression(preds[0],
                                    self.args.conf,
                                    self.args.iou,
                                    agnostic=self.args.agnostic_nms,
                                    max_det=self.args.max_det,
                                    nc=len(self.model.names),
                                    classes=self.args.classes)
        # print(self.args.agnostic_nms)
        # print(self.args.conf)
        # print(self.args.iou)
        # print(self.args.max_det)
        # print(len(self.model.names))
        # print(self.args.classes)
        # exit(1)
        results = []
        print(preds[0].shape)
        print(p[0].shape)
        exit(1)
        print(img.shape)
        print(orig_imgs[0].shape)
        exit(1)
        print(preds.shape)
        proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # second output is len 3 if pt, but only 1 if exported
        for i, pred in enumerate(p):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            path = self.batch[0]
            img_path = path[i] if isinstance(path, list) else path
            if not len(pred):  # save empty boxes
                results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6]))
                continue
            if self.args.retina_masks:
                if not isinstance(orig_imgs, torch.Tensor):
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
            else:
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
                if not isinstance(orig_imgs, torch.Tensor):
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(
                Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks))
        return results


def predict(cfg, use_python=False):
    """Runs YOLO object detection on an image or video source."""
    model = cfg["model"] 
    source = cfg["source"] 

    args = dict(model=model, source=source)
    if use_python:
        from ultralytics import YOLO
        YOLO(model)(**args)
    else:
        predictor = SegmentationPredictor(overrides=args)
        print(predictor)
        predictor.predict_cli()

def overlay(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.
    https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined

import matplotlib.pyplot as plt
from PIL import Image

def fast_show_mask(
    annotation,
    ax,
    random_color=False,
    bbox=None,
    retinamask=True,
    target_height=960,
    target_width=960,
):
    mask_sum = annotation.shape[0]
    height = annotation.shape[1]
    weight = annotation.shape[2]
    print("[Info] Size of mask: ", (weight, height))
    # 将annotation 按照面积 排序
    areas = np.sum(annotation, axis=(1, 2))
    sorted_indices = np.argsort(areas)[::1]
    annotation = annotation[sorted_indices]

    index = (annotation != 0).argmax(axis=0)
    if random_color == True:
        color = np.random.random((mask_sum, 1, 1, 3))
    else:
        color = np.ones((mask_sum, 1, 1, 3)) * np.array([30 / 255, 144 / 255, 255 / 255])
    transparency = np.ones((mask_sum, 1, 1, 1)) * 0.6
    visual = np.concatenate([color, transparency], axis=-1)
    mask_image = np.expand_dims(annotation, -1) * visual

    mask = np.zeros((height, weight, 4))

    h_indices, w_indices = np.meshgrid(np.arange(height), np.arange(weight), indexing='ij')
    indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))

    mask[h_indices, w_indices, :] = mask_image[indices]
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='b', linewidth=1))

    if retinamask == False:
        print("Resize")
        mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    print("[Info] Maks resized: ", mask.shape)
    return mask

def fast_process(
    annotations,
    image_origin,
    image,
    device,
    scale,
    better_quality=False,
    mask_random_color=True,
    bbox=None,
    use_retina=True,
    withContours=True,
    ):
    if isinstance(annotations[0], dict):
        annotations = [annotation['segmentation'] for annotation in annotations]

    original_w, original_h = image.size
    print("[Info] Origin shape: ", (original_w, original_h))
    if better_quality:
        if isinstance(annotations[0], torch.Tensor):
            annotations = np.array(annotations.cpu())
        for i, mask in enumerate(annotations):
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            annotations[i] = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((8, 8), np.uint8))

    annotations = np.array(annotations)
    inner_mask = fast_show_mask(
        annotations,
        plt.gca(),
        random_color=mask_random_color,
        bbox=bbox,
        retinamask=use_retina,
        target_height=original_h,
        target_width=original_w,
    )
    if isinstance(annotations, torch.Tensor):
        annotations = annotations.cpu().numpy()

    if withContours:
        contour_all = []
        temp = np.zeros((original_h, original_w, 1))
        for i, mask in enumerate(annotations):
            if type(mask) == dict:
                mask = mask['segmentation']
            annotation = mask.astype(np.uint8)
            if use_retina == False:
                annotation = cv2.resize(
                    annotation,
                    (original_w, original_h),
                    interpolation=cv2.INTER_NEAREST,
                )
            contours, _ = cv2.findContours(annotation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour_all.append(contour)
        cv2.drawContours(temp, contour_all, -1, (255, 255, 255), 2 // scale)
        color = np.array([0 / 255, 0 / 255, 255 / 255, 0.9])
        contour_mask = temp / 255 * color.reshape(1, 1, -1)

    image = image.convert('RGBA')
    overlay_inner = Image.fromarray((inner_mask * 255).astype(np.uint8), 'RGBA')
    image.paste(overlay_inner, (0, 0), overlay_inner)

    if withContours:
        overlay_contour = Image.fromarray((contour_mask * 255).astype(np.uint8), 'RGBA')
        image.paste(overlay_contour, (0, 0), overlay_contour)

    return image

def segment_everything(
    input,
    results,
    input_size=1024, 
    iou_threshold=0.7,
    conf_threshold=0.25,
    better_quality=False,
    withContours=True,
    use_retina=True,
    mask_random_color=True,
    device = "cuda"
    ):
    input_size = int(input_size)  # 确保 imgsz 是整数
    print("[Info] Truoc resize: ", input.size)
    input_ori = input.copy()
    # Thanks for the suggestion by hysts in HuggingFace.
    w, h = input.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    input = input.resize((new_w, new_h))

    print("[Info] Sau resize: ", input.size)

    # exit(1)
    fig = fast_process(annotations=results[0].masks.data,
                        image=input,
                        image_origin= input,
                        device=device,
                        scale=(1024 // input_size),
                        better_quality=better_quality,
                        mask_random_color=mask_random_color,
                        bbox=None,
                        use_retina=use_retina,
                        withContours=withContours,)
    return fig


if __name__ == "__main__":
  cfg = {"model": "/models/ultralytics/yolov8s-seg.onnx", "source": "/models/FastSam/cat.jpg"}
  predict(cfg)
  