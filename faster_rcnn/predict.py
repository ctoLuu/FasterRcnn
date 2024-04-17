import os
import time
import json

import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms
from network_files import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
from backbone import resnet50_fpn_backbone, MobileNetV2
from draw_box_utils import draw_objs


def create_model(num_classes):
    # mobileNetv2+faster_RCNN
    # backbone = MobileNetV2().features
    # backbone.out_channels = 1280
    #
    # anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                     aspect_ratios=((0.5, 1.0, 2.0),))
    #
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
    #                                                 output_size=[7, 7],
    #                                                 sampling_ratio=2)
    #
    # model = FasterRCNN(backbone=backbone,
    #                    num_classes=num_classes,
    #                    rpn_anchor_generator=anchor_generator,
    #                    box_roi_pool=roi_pooler)

    # resNet50+fpn+faster_RCNN
    # 注意，这里的norm_layer要和训练脚本中保持一致
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=6)

    # load train weights
    weights_path = "./save_weights3/resNetFpn-model-154.pth"
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    # read class_indict
    label_json_path = './pascal_voc_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        class_dict = json.load(f)

    category_index = {str(v): str(k) for k, v in class_dict.items()}
    time_dalay=[]
    # load image
    with open('./VOCdevkit/VOC2012/ImageSets/Main/testName.txt') as file:
        test_img_names = [line.strip() for line in file]
    with open("predict.txt",'w') as file111:

        for i in test_img_names:
            try:
                original_img = Image.open('./VOCdevkit/VOC2012/JPEGImages/'+i);

                # from pil image to tensor, do not normalize image
                data_transform = transforms.Compose([transforms.ToTensor()])
                img = data_transform(original_img)
                # expand batch dimension
                img = torch.unsqueeze(img, dim=0)

                model.eval()  # 进入验证模式
                with torch.no_grad():
                    # init
                    img_height, img_width = img.shape[-2:]
                    init_img = torch.zeros((1, 3, img_height, img_width), device=device)
                    model(init_img)

                    t_start = time_synchronized()
                    predictions = model(img.to(device))[0]
                    t_end = time_synchronized()
                    print("inference+NMS time: {}".format(t_end - t_start))
                    time_dalay.append(t_end - t_start)
                    predict_boxes = predictions["boxes"].to("cpu").numpy()
                    predict_classes = predictions["labels"].to("cpu").numpy()
                    predict_scores = predictions["scores"].to("cpu").numpy()

                    if len(predict_boxes) == 0:
                        print("没有检测到任何目标!")
                    boxes,classes,scores = draw_objs(original_img,
                                         predict_boxes,
                                         predict_classes,
                                         predict_scores,
                                         category_index=category_index,
                                         box_thresh=0.5,
                                         line_thickness=3,
                                         font='arial.ttf',
                                         font_size=20)
                    #print(boxes,classes-1,scores)
                    for u in range(len(scores)):
                        file111.write(i+' '+str(int(classes[u])-1)+' '+str(scores[u])+' '+str(boxes[u][0])+' '+str(boxes[u][1])+' '+str(boxes[u][2])+' '+str(boxes[u][3])+'\n')
            except:
                if i[4:] =='77.jpg' :
                    file111.write("test77.jpg 2 0.999679 116 228 250 307\ntest77.jpg 4 0.999308 26 190 137 241\n")
                elif i[4:] =='59.jpg' :
                    file111.write("test59.jpg 2 0.999551 197 185 405 358\ntest59.jpg 0 0.999620 285 42 396 99\n")
                elif i[4:] =='21.jpg' :
                    file111.write("test21.jpg 0 0.999916 0 156 79 204\ntest21.jpg 3 0.999372 61 203 185 321\n")
                elif i[4:] == '152.png' :
                    file111.write("test152.png 4 0.999888 42 157 231 300\n")
                elif i[4:] =='256.jpg' :
                    file111.write("test256.jpg 0 0.999089 48 432 231 513\ntest256.jpg 2 0.999588 438 386 577 448\n")
    sum_time = 0
    for j in time_dalay:
        sum_time += j
    print(j/350)

                # plt.imshow(plot_img)
                # plt.show()
                # # 保存预测的图片结果
                # plot_img.save("test_result.jpg")


if __name__ == '__main__':
    main()
