from ultralytics import YOLO
# from utils.tools import *
import argparse
# from models.experimental import attempt_load
import torch.nn as nn
import torch
import onnx
import onnx_graphsurgeon as gs
# from models.yolo import SegmentationModel
import ast


class FastSamAddNMS(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()

    def forward(self, input):
        """ 
            Split output [n_batch, 84, n_bboxes] to 3 output: bboxes, scores, classes
        """ 
        # x, y, w, h -> x1, y1, x2, y2
        output = self.model(input)
        print('Output: ', len(output))
        # print(len(output[1]))
        # print(output[1].shape)
        # print(output[0].shape)
        # for x in output:
        #     if type(x).__name__ == 'tuple':
        #         print([y.shape for y in x])
        #     else:
        #         print('single ', x.shape)
        # exit(1)
        output = output[0]
        print(output.shape)
        exit(1)
        output = output.permute(0, 2, 1)
        print(output[0][0])
        print("[INFO] Output's origin model shape: ", output.shape)
        bboxes_x = output[..., 0:1]
        bboxes_y = output[..., 1:2]
        bboxes_w = output[..., 2:3]
        bboxes_h = output[..., 3:4]
        bboxes_x1 = bboxes_x - bboxes_w/2
        bboxes_y1 = bboxes_y - bboxes_h/2
        bboxes_x2 = bboxes_x + bboxes_w/2
        bboxes_y2 = bboxes_y + bboxes_h/2
        bboxes = torch.cat([bboxes_x1, bboxes_y1, bboxes_x2, bboxes_y2], dim = -1)
        bboxes = bboxes.unsqueeze(2) # [n_batch, n_bboxes, 4] -> [n_batch, n_bboxes, 1, 4]
        obj_conf = output[..., 4:]
        scores = obj_conf
        # cls_conf = output[..., 5:]
        # scores   = obj_conf * cls_conf # conf = obj_conf * cls_conf
        print(scores)
        print(bboxes)
        return bboxes, scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='/data/disk1/hungpham/FastSAM/weights/FastSAM-x.pt', help='weights path')
    # parser.add_argument('--cfg', type=str, default='cfg/yolo_nas.cfg', help='config path')
    parser.add_argument('--output', type=str, default='/data/disk1/hungpham/FastSAM/weights/', help='output ONNX model path')
    parser.add_argument('--max_size', type=int, default=416, help='max size of input image')
    opt = parser.parse_args()

    # model_cfg = opt.cfg
    model_weights = opt.weights
    output_model_path = opt.output
    max_size = opt.max_size
    device = torch.device("cuda")

    # load model 
    print("[Info] Load Model")
    # model = attempt_load(model_weights, device=device, inplace=True, fuse=True)
    model_ = YOLO(model_weights)
    # print(model_.__dict__)
    model = model_.model
    # print(model.__dict__)
    # exit(1)
    # print(model.shape)
    # print(type(model))
    # exit(1)

    img = torch.zeros(1, 3, max_size, max_size).to(device)

    # results = model_(
    #     "/data/disk1/hungpham/FastSAM/images/cat.jpg",
    #     imgsz=max_size,
    #     device=device,
    #     retina_masks=True,
    #     iou=0.9,
    #     conf=0.4,
    #     max_det=100,
    # )
    # print(results[0].masks.data.shape)
    # exit(1)

    print("[Info] Preprocess Model")
    # model = FastSamAddNMS(model)
    # exit(1)
    output_names = ['output0', 'output1'] #if isinstance(model, SegmentationModel) else ['output0']
    dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1,3,640,640)
    dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
    dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)
    # if isinstance(model, SegmentationModel):
    #     dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
    #     dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)

    model.eval().to(device)

    
    # print(img)

    # for _ in range(2):
    #     y = model(img)  # dry runs
    print('[INFO] Convert from Torch to ONNX')

    # model_path = "/data/disk1/hungpham/FastSAM/weights/FastSAM-x.pt"
    # model = YOLO(model_weights)

    # model.to(device).eval()

    torch.onnx.export(model,               # model being run
                    img,                         # model input (or a tuple for multiple inputs)
                    output_model_path,   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=11,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['images'],   # the model's input names
                    output_names = output_names, # the model's output names
                    dynamic_axes=dynamic)

    print('[INFO] Finished Convert!')