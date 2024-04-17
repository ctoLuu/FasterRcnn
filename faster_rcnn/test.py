import torch
import torchvision.transforms as transforms
import numpy as np
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from PIL import Image
import cv2

# Load the model
num_classes = 5
from backbone import resnet50_fpn_backbone  # 导入自定义的 ResNet-50 FPN backbone

backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
model = FasterRCNN(backbone=backbone, num_classes=num_classes + 1)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Load the model weights
model_PATH = './save_weights/resNetFpn-model-106.pth'
checkpoint = torch.load(model_PATH, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)

model.eval()

# List of test image paths
with open('image_paths.txt', 'r') as file:
    test_img_paths = [line.strip() for line in file]

results = []

for test_img_path in test_img_paths:
    test_img = Image.open(test_img_path).convert('RGB')
    test_img_array = np.array(test_img)
    transform = transforms.Compose([transforms.ToTensor()])
    test_tensor = transform(test_img_array)

    pred = model([test_tensor])
    boxes = pred[0]['boxes']
    labels = pred[0]['labels']
    scores = pred[0]['scores']

    boxes_ = []
    labels_ = []
    scores_ = []

    for box in boxes:
        xmin, ymin, xmax, ymax = box
        xmin_, ymin_, xmax_, ymax_ = int(xmin.item()), int(ymin.item()), int(xmax.item()), int(ymax.item())
        boxes_.append([xmin_, ymin_, xmax_, ymax_])

    for label in labels:
        labels_.append(label.item())

    for score in scores:
        scores_.append(score.item())

    results.append({'image_path': test_img_path, 'boxes': boxes_, 'labels': labels_, 'scores': scores_})

# Save results to a text file
with open('results.txt', 'w') as file:
    for result in results:
        file.write(f"Image Path: {result['image_path']}\n")
        for i, box in enumerate(result['boxes']):
            file.write(f"Box: {box}, Class: {result['labels'][i]}, Score: {result['scores'][i]}\n")
        file.write('\n')

# Display the results
for result in results:
    pred_rec = cv2.imread(result['image_path'])
    for i, (xmin_, ymin_, xmax_, ymax_) in enumerate(result['boxes']):
        pred_rec = cv2.rectangle(pred_rec, (xmin_, ymin_), (xmax_, ymax_), color=(255, 0, 0), thickness=4)
        cv2.putText(pred_rec, f"Class: {result['labels'][i]}, Score: {result['scores'][i]}", (xmin_, ymin_),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Predict Result!', pred_rec)
    cv2.waitKey()

cv2.destroyAllWindows()
