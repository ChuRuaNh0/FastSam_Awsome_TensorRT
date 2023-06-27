import os
import cv2
import numpy as np
import time
import logging
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda 
from .trt_loader import TrtModel

# class HostDeviceMem(object):
#     def __init__(self, host_mem, device_mem):
#         self.host = host_mem
#         self.device = device_mem

#     def __str__(self):
#         return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

#     def __repr__(self):
#         return self.__str__()

# def alloc_buf(engine):
#     inputs = []
#     outputs = []
#     bindings = []
#     stream = cuda.Stream()
#     for binding in engine:
#         size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
#         dtype = trt.nptype(engine.get_binding_dtype(binding))
#         # Allocate host and device buffers
#         host_mem = cuda.pagelocked_empty(size, dtype)
#         device_mem = cuda.mem_alloc(host_mem.nbytes)
#         # Append the device buffer to device bindings.
#         bindings.append(int(device_mem))
#         # Append to the appropriate list.
#         if engine.binding_is_input(binding):
#             inputs.append(HostDeviceMem(host_mem, device_mem))
#         else:
#             outputs.append(HostDeviceMem(host_mem, device_mem))
#     return inputs, outputs, bindings, stream

# def inference(context, bindings, inputs, outputs, stream):
#     # Transfer input data to the GPU.
#     [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
#     # Run inference.
#     context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
#     # Transfer predictions back from the GPU.
#     [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
#     # Synchronize the stream
#     stream.synchronize()
#     # Return only the host outputs.
#     return [out.host for out in outputs]

def alloc_buf(engine):
    # host cpu mem
    h_in_size = trt.volume(engine.get_binding_shape(0))
    h_out_size = trt.volume(engine.get_binding_shape(1))
    h_in_dtype = trt.nptype(engine.get_binding_dtype(0))
    h_out_dtype = trt.nptype(engine.get_binding_dtype(1))
    in_cpu = cuda.pagelocked_empty(h_in_size, h_in_dtype)
    out_cpu = cuda.pagelocked_empty(h_out_size, h_out_dtype)
    # allocate gpu mem
    in_gpu = cuda.mem_alloc(in_cpu.nbytes)
    out_gpu = cuda.mem_alloc(out_cpu.nbytes)
    stream = cuda.Stream()
    return in_cpu, out_cpu, in_gpu, out_gpu, stream

def inference(engine, context, inputs, out_cpu, in_gpu, out_gpu, stream):
    # sync version
    cuda.memcpy_htod(in_gpu, inputs)
    context.execute(1, [int(in_gpu), int(out_gpu)])
    cuda.memcpy_dtoh(out_cpu, out_gpu)
    return out_cpu

class Arcface:

    def __init__(self, rec_name: str='/models/trt-engines/arcface_r100_v1/arcface_r100_v1.plan'):
        self.rec_model = TrtModel(rec_name)
        self.input_shape = None
        self.max_batch_size = 1

    # warmup
    def prepare(self, ctx_id=0):
        logging.info("Warming up ArcFace TensorRT engine...")
        self.rec_model.build()
        self.input_shape = self.rec_model.input_shapes[0]
        self.max_batch_size = self.rec_model.max_batch_size
        if self.input_shape[0] == -1:
            self.input_shape = (1,) + self.input_shape[1:]

        self.rec_model.run(np.zeros(self.input_shape, np.float32))
        logging.info(f"Engine warmup complete! Expecting input shape: {self.input_shape}. Max batch size: {self.max_batch_size}")

    def get_embedding(self, face_img):

        if not isinstance(face_img, list):
            face_img = [face_img]

        for i, img in enumerate(face_img):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2, 0, 1))
            #img = np.expand_dims(img, axis=0)
            face_img[i] = img
        #assert face_img.shape == self.rec_model.input_shapes[0]
        face_img = np.stack(face_img)
        embeddings = self.rec_model.run(face_img, deflatten=True)[0]
        return embeddings


class FaceGenderage:

    def __init__(self, rec_name: str='/models/trt-engines/genderage_v1/genderage_v1.plan'):
        self.rec_model = TrtModel(rec_name)
        self.input_shape = None

    # warmup
    def prepare(self, ctx_id=0):
        logging.info("Warming up GenderAge TensorRT engine...")
        self.rec_model.build()
        self.input_shape = self.rec_model.input_shapes[0]
        self.rec_model.run(np.zeros(self.input_shape, np.float32))
        logging.info(f"Engine warmup complete! Expecting input shape: {self.input_shape}")

    def get(self, face_img):
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = np.transpose(face_img, (2, 0, 1))
        face_img = np.expand_dims(face_img, axis=0)
        assert face_img.shape == self.rec_model.input_shapes[0]
        ret = self.rec_model.run(face_img, deflatten=True)[0]
        g = ret[:, 0:2].flatten()
        gender = np.argmax(g)
        a = ret[:, 2:202].reshape((100, 2))
        a = np.argmax(a, axis=1)
        age = int(sum(a))
        return gender, age

class DetectorInfer:

    def __init__(self, model='/models/trt-engines/centerface/centerface.plan',
                 output_order=None):
        self.rec_model = TrtModel(model)
        self.model_name = os.path.basename(model)
        self.input_shape = None
        self.output_order = output_order

    # warmup
    def prepare(self, ctx_id=0):
        logging.info(f"Warming up face detector TensorRT engine...")
        self.rec_model.build()
        self.input_shape = self.rec_model.input_shapes[0]
        if not self.output_order:
            self.output_order = self.rec_model.out_names
        self.rec_model.run(np.zeros(self.input_shape, np.float32))
        logging.info(f"Engine warmup complete! Expecting input shape: {self.input_shape}")

    def run(self, input):
        net_out = self.rec_model.run(input, deflatten=True, as_dict=True)
        net_out = [net_out[e] for e in self.output_order]
        return net_out