import onnxruntime as onnxrt
from dataset import *
import cv2
import numpy as np
from utils import *
import logging
from configs import categories

logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)

provider = 'CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider'
print("provider:", provider)
onnx_session = onnxrt.InferenceSession(
        "models/model_v0.onnx", providers=[provider])
logging.info("Load model ONNX done.")


categories_ = categories

class Opt():
    def __init__(self):
        self.root_data = ""
        self.imgH = 700
        self.imgW = 1000

opt = Opt()

def area_calc(bbox, imgW, imgH):
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    area = (x2 - x1) * (y2 - y1)
    if area / (imgW * imgH) < 9e-4:
        return False
    return True

def inference(images):
    batch_image = []
    dt = DataTransformer(opt)

    for image in images:
        color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(color_coverted)
        image = dt(image)
        width, height = image.shape[2], image.shape[1]
        image = image.detach().numpy()
        batch_image.append(image)

    batch_image = np.array(batch_image).astype(np.float32)

    categories = categories_
    
    input_name = onnx_session.get_inputs()[0].name
    pred = onnx_session.run(["P", "T"], {input_name: batch_image})
    P, T = pred[0], pred[1]

    binary_map = P >= T
    num_batch = binary_map.shape[0]
    post_binary_map = [[m for m in binary_map[index].astype(
        'uint8')] for index in range(num_batch)]
    pred_bboxes = [multi_apply(post_process, post_binary_map[index])
                   for index in range(num_batch)]
    map = []
    for index in range(num_batch):
        pred_bboxes_new = []
        for i, bboxes in enumerate(pred_bboxes[index]):
            if len(bboxes) == 0:
                pred_bboxes_new.append([])
            else:
                tmp = []
                for bbox in bboxes:
                    if area_calc(bbox, width, height):
                        tmp.append([int(i) for i in bbox])
                pred_bboxes_new.append(tmp)
        map.append(dict(zip(categories, pred_bboxes_new)))

    batch_image = torch.from_numpy(batch_image)
    image = unnormalize(batch_image, mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225))
    image = image.permute(0, 2, 3, 1).cpu().numpy()
    image = (image * 255).astype('uint8').copy()

    return map, image
