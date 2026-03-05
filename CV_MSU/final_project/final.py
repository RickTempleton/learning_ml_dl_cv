import cv2
import numpy as np
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image



model = EfficientNet.from_name('efficientnet-b0', num_classes=4)
checkpoint = torch.load('best.pth')
model.load_state_dict(checkpoint['model'], strict=False)

model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

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
output_video = cv2.VideoWriter('result_video.mp4', fourcc, 30.0, (frame_width, frame_height))

while True:
    ok, frame = video.read()
    if not ok:
        break

    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    graybackground = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(grayFrame, graybackground)

    diff = cv2.blur(diff, (4, 4)) 
    retval, diff = cv2.threshold(diff, 55, 255, cv2.THRESH_BINARY)

    contours, h = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    

    for el in contours:
        x, y, w, h = cv2.boundingRect(el)
        if (w * h > 50) and (w < 2 * h) and (w * 2 > h): 
            
            roi = frame[y:y+h, x:x+w]

            roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            roi_transformed = transform(roi_pil).unsqueeze(0)  

            with torch.no_grad():
                outputs = model(roi_transformed)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, dim=1)

            class_id = predicted_class.item()
            probability = confidence.item()

            label = f'Class: {class_id}, Prob: {probability:.2f}'
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 1)

    output_video.write(frame)

video.release()
output_video.release()
cv2.destroyAllWindows()
