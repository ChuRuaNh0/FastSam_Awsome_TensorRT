/usr/src/tensorrt/bin/trtexec --onnx=/models/FastSam/fast_sam_1024.onnx \
                                --saveEngine=/models/fastSAm_wrapper/fast_sam_1024.trt \
                                --explicitBatch \
                                --minShapes=images:1x3x1024x1024 \
                                --optShapes=images:1x3x1024x1024 \
                                --maxShapes=images:4x3x1024x1024 \
                                --verbose \
                                --device=0