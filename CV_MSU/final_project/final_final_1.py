import cv2
import numpy as np
# import torch
# from torchvision import transforms
# from efficientnet_pytorch import EfficientNet
from PIL import Image
# import torch.nn as nn
from collections import OrderedDict
import time
import matplotlib.pyplot as plt

# checkpoint = torch.load('best_3.pth', map_location=torch.device('cpu'))
# model = EfficientNet.from_pretrained('efficientnet-b0')
# model._fc = nn.Sequential(
#     nn.Dropout(0.2),
#     nn.Linear(model._fc.in_features, 4), 
#     nn.Softmax(dim=1)
# )

# state_dict = checkpoint['model']
# model_dict = model.state_dict()
# new_state_dict = OrderedDict()
# matched_layers, discarded_layers = [], []
# for k, v in state_dict.items():
#     if k.startswith('module.'):
#         k = k[7:]  
        
#     if k in model_dict and model_dict[k].size() == v.size():
#         new_state_dict[k] = v
#         matched_layers.append(k)
#     else:
#         discarded_layers.append(k)

# model_dict.update(new_state_dict)
# model.load_state_dict(model_dict)
# model.eval()


# class_names = ['bus', 'car', 'motorcycle', 'truck']

# def get_transform():
#     return transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

# def preprocess_frame(frame):
#     transform = get_transform()
#     frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     return transform(frame_pil).unsqueeze(0)

# def classify_frame(frame):
#     input_tensor = preprocess_frame(frame)
#     with torch.no_grad():
#         outputs = model(input_tensor)
#         probabilities = torch.softmax(outputs, dim=1)
#         confidence, predicted_class = torch.max(probabilities, dim=1)
#     return predicted_class.item(), confidence.item()


def last_object(element_1, contours):
    eps = 200
    rez = element_1
    x_1, y_1, w_1, h_1 = cv2.boundingRect(element_1)
    min = -1
    for element_2 in contours:
        x_2, y_2, w_2, h_2 = cv2.boundingRect(element_2)
        sdvig = (x_1 -x_2)**2 + (y_1 -y_2)**2 + (w_1 -w_2)**2 + (h_1 -h_2)**2
        if sdvig < eps:
            if min > 0:
                if sdvig < min:
                    rez = element_2
                    min = sdvig
            else:
                min = sdvig
                rez = element_2
    return rez








video = cv2.VideoCapture("road.mp4")
background = cv2.imread("background.png")

ok, frame1 = video.read()
if not ok:
    print("Ошибка при чтении видео")
    video.release()
    cv2.destroyAllWindows()
    exit()


frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))


fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
output_video = cv2.VideoWriter('result_video_1.mp4', fourcc, 30.0, (frame_width, frame_height))

contours_last = []
count = 0
while True:
# while count <60:
#     count = count + 1
    ret, frame = video.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_frame, gray_background)


    diff = cv2.blur(diff, (4, 4)) 
    retval, diff = cv2.threshold(diff, 55, 255, cv2.THRESH_BINARY)


    contours, h = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    


    for el in contours:
        x, y, w, h = cv2.boundingRect(el)
        if (w * h > 50) and (w < 2 * h) and (w * 2 > h):  
            roi = frame[y:y+h, x:x+w]
            # class_id, probability = classify_frame(roi)

            if contours_last != []:
                elem_last= last_object(el, contours_last)
                x_2, y_2, w_2, h_2 = cv2.boundingRect(elem_last)
                # print(f"{y} {y_2}")
                if y_2 > y:
                    cv2.putText(frame, "away", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    brightness = 50
                    contrast = 30
                    roi = np.int16(roi)
                    roi = roi * (contrast/127+1) - contrast + brightness
                    roi = np.clip(roi, 0, 255)
                    roi = np.uint8(roi)
                    frame[y:y+h, x:x+w] = roi
                # if y_2 < y:
                #     cv2.putText(frame, "approach", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            # label = f'Class: {class_names[class_id]}, Prob: {probability:.2f}'
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.5, (255, 0, 0), 1)



    contours_last = contours

    output_video.write(frame)

video.release()
output_video.release()
cv2.destroyAllWindows()
