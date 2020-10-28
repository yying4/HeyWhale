import os
import re
import shutil
import pandas as pd

imgfile_path = './sequences'
answer_path = './annotations'

# 按照文件夹径读取文件夹内文件名并存入一个list
def readfile(filePath):
    name = list(os.listdir(filePath))
    return name

# 提取图像文件名和答案文件名
img_folder = readfile(imgfile_path)
answer_file = readfile(answer_path)

# i是文件夹中的每一个图像文件夹，例如‘uav0000013_00000_v’
for i in img_folder:

    # 读取每一图像文件夹对应的txt文件，例如‘uav0000013_00000_v.txt’
    answer = pd.read_csv(answer_path + './' + i + '.txt', sep = ',',header=None)

    # 创建新的train和test文件夹准备存储5-5分的图片
    os.mkdir("./sequences_train./" + i + '_train')
    os.mkdir("./sequences_test./" + i + '_test')

    # 每一图片名，例如‘0000001.jpg’
    imgs = readfile(imgfile_path + './' + i)

    # 每一图片名所对应的桢数，例如‘0000001’对应帧数‘1’
    frame = [int(i[:-4]) for i in imgs]

    # 根据每一文件夹内最大图片数算出一半界限，前一半作为训练图片（训练帧数），后一半作为测试图片（测试帧数）
    number = max([int(i[:-4]) for i in imgs])
    middle = (number//4)*3
    img_train = imgs[:middle]
    img_test = imgs[middle:]
    frame_train = frame[:middle]
    frame_test = frame[middle:]

    # 根据分出的训练帧数提取txt内相应的数据，txt第一列表示帧数
    train_list = []
    for p in range(len(answer.iloc[:,0])):
        if answer.iloc[p,0] in frame_train:
            train_list.append(p)
    answer_train = answer.iloc[train_list,:]

    # 根据分出的测试帧数提取txt内相应的数据，txt第一列表示帧数
    test_list = []
    for p in range(len(answer.iloc[:,0])):
        if answer.iloc[p,0] in frame_test:
            test_list.append(p)
    answer_test = answer.iloc[test_list,:]

    # 将提取的答案数据保存到新的对应训练和测试文件夹中，以txt文件存储
    answer_train.to_csv('annotations_train./' + i + '_train' + '.txt', sep=',', index=False, header = None)
    answer_test.to_csv('annotations_test./' + i + '_test' + '.txt', sep=',', index=False, header = None)
    
    # 将分离的使用图片保存到新的对应训练和测试文件夹中
    for j in img_train:
        shutil.copy(imgfile_path + './'+ i +'./' + j, './sequences_train./' + i + '_train' +'./' + j)

    for j in img_test:
        shutil.copy(imgfile_path + './'+ i +'./' + j, './sequences_test./'+ i + '_test' +'./' + j)

# 全部完成
print('Finished!')


