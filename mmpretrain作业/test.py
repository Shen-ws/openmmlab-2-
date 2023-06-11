from mmpretrain import ImageClassificationInferencer

inferencer = ImageClassificationInferencer('./configs/resnet/resnet18_fintune.py',
                                           pretrained='./work_dirs/resnet18_fintune/epoch_10.pth')

image_list = ['grape.jpg']

# 单独对每张图片预测

result0 = inferencer(image_list[0], show=True)




