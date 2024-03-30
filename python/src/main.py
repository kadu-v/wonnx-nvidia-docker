import onnx
import wonnx
from onnx import helper
from onnx import TensorProto
from torchvision import transforms
import numpy as np
import cv2
import os
import time

basedir = os.path.dirname(os.path.realpath(__file__))

def test_squeezenet():
    image = cv2.imread(os.path.join(basedir, "../../data/images/pelican.jpeg"))
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # apply transforms to the input image
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    input_tensor = transform(rgb_image)

    # Create the model (ModelProto)
    session = wonnx.Session.from_path(
        os.path.join(basedir, "../../data/models/opt-squeeze.onnx")
    )
    inputs = {"data": input_tensor.flatten().tolist()}
    
    start = time.time()
    result = session.run(inputs)["squeezenet0_flatten0_reshape0"]
    end = time.time()

    print(f"result 144={result[144]} argmax={np.argmax(result)} score_max={np.max(result)} time={(end - start) * 1000}ms")

    assert (
        np.argmax(result) == 144
    ), "Squeezenet does not work"

if __name__ == '__main__':
    test_squeezenet()