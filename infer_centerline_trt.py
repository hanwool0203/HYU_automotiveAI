import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

ENGINE_PATH = "centerline_unet_mobilenetv2_fp16.engine"
IMAGE_PATH = "test.jpg"

INPUT_W = 320
INPUT_H = 180

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def preprocess(image_bgr):
    image = cv2.resize(image_bgr, (INPUT_W, INPUT_H))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image.astype(np.float32) / 255.0
    image = (image - MEAN) / STD

    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    image = np.expand_dims(image, axis=0)   # CHW -> NCHW

    return np.ascontiguousarray(image.astype(np.float32))


def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        raise RuntimeError("TensorRT engine 로드 실패")

    return engine


def get_tensor_names(engine):
    input_names = []
    output_names = []

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)

        if mode == trt.TensorIOMode.INPUT:
            input_names.append(name)
        elif mode == trt.TensorIOMode.OUTPUT:
            output_names.append(name)

    return input_names, output_names


def main():
    engine = load_engine(ENGINE_PATH)
    context = engine.create_execution_context()

    input_names, output_names = get_tensor_names(engine)

    print("input_names :", input_names)
    print("output_names:", output_names)

    input_name = input_names[0]
    output_name = output_names[0]

    image_bgr = cv2.imread(IMAGE_PATH)
    if image_bgr is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {IMAGE_PATH}")

    inp = preprocess(image_bgr)

    # Dynamic shape engine일 경우를 대비
    input_shape = tuple(engine.get_tensor_shape(input_name))
    print("engine input shape:", input_shape)

    if -1 in input_shape:
        context.set_input_shape(input_name, inp.shape)

    output_shape = tuple(context.get_tensor_shape(output_name))
    print("actual input shape :", inp.shape)
    print("output shape       :", output_shape)

    output_dtype = trt.nptype(engine.get_tensor_dtype(output_name))
    out = np.empty(output_shape, dtype=output_dtype)

    # GPU 메모리 할당
    d_input = cuda.mem_alloc(inp.nbytes)
    d_output = cuda.mem_alloc(out.nbytes)

    stream = cuda.Stream()

    # TensorRT 10 방식: tensor address 직접 지정
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    # Host -> Device
    cuda.memcpy_htod_async(d_input, inp, stream)

    # Inference
    ok = context.execute_async_v3(stream_handle=stream.handle)
    if not ok:
        raise RuntimeError("TensorRT execute_async_v3 실패")

    # Device -> Host
    cuda.memcpy_dtoh_async(out, d_output, stream)
    stream.synchronize()

    print("output min/max:", float(out.min()), float(out.max()))

    # 네 모델 output: [1, 1, 180, 320] logits
    logits = out[0, 0]
    prob = sigmoid(logits)
    mask = (prob > 0.5).astype(np.uint8) * 255

    cv2.imwrite("centerline_mask.png", mask)

    overlay = cv2.resize(image_bgr, (INPUT_W, INPUT_H)).copy()
    overlay[mask > 0] = (0, 0, 255)

    cv2.imwrite("centerline_overlay.png", overlay)

    print("saved: centerline_mask.png")
    print("saved: centerline_overlay.png")


if __name__ == "__main__":
    main()