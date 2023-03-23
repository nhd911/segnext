from run_onnx import inference
import numpy
import cv2
import matplotlib.pyplot as plt
import numpy as np


class DetectorDriving():
    def __init__(self) -> None:
        super().__init__()

    def detect_fields(self, cropped_image: numpy.ndarray):
        if cropped_image is not None:
            preds, images = inference(cropped_image)
            
            pred = preds[0]
            image = images[0]
            font = cv2.FONT_HERSHEY_SIMPLEX

            fontScale = 0.3

            # Blue color in BGR
            color = (255, 0, 0)

            for i in pred:
                if len(pred[i]) > 0:
                    for (x1, y1, x2, y2) in pred[i]:
                        cv2.putText(image, i, (x1, y1-5), font,
                    fontScale, color, 1, cv2.LINE_AA)
                        cv2.rectangle(image, (x1-4, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite("1.jpg", image)
            img0 = image
            
        if preds is not None:

            output_results = []
            new_preds = []
            for index, pred in enumerate(preds):
                boxes = {}
                new_pred = {}
                for label, box in pred.items():
                    new_label = label
                    if len(box) != 0:

                        if new_label not in boxes:
                            boxes[new_label] = []
                        for b in box:
                            boxes[new_label].append({
                                "top_left_y": round(b[1]),
                                "bot_right_y": round(b[3]),
                                "top_left_x": round(b[0]),
                                "bot_right_x": round(b[2])
                            })
                        if new_label not in new_pred:
                            new_pred[new_label] = []

                        new_pred[new_label] += box
                new_preds.append(new_pred)

                output_result = {}
                for label in boxes:
                    img_arr = []

                    for item in boxes[label]:
                        cropped__img = images[index][
                            item["top_left_y"]:item["bot_right_y"],
                            item["top_left_x"]:item["bot_right_x"]
                        ]
                        img_arr.append(cropped__img)
                    output_result[label] = img_arr

                output_results.append(output_result)

            return output_results, new_preds


if __name__ == "__main__":
    detector = DetectorDriving()
    image = cv2.imread("dataset/val/data/28.jpg")
    output_results, preds = detector.detect_fields(
        np.array([image]))
    print(output_results, preds)