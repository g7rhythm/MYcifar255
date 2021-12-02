# MYcifar255 简介
1，根据tensorflow -cifar10 示例 改进，以便适应更多图片与分类。2，完善打包图片到Bin文件的生成机制

# Deployment
python 3.5

numpy==1.12.0

tensorflow==1.2.1

# 使用步骤
第一步：下载训练数据集  （200多种图片分类，图片来源：ILSVRC2012，图片压缩过）
链接：https://pan.baidu.com/s/1lbM9-qs1BA2rzRFviZ3t6A （2021.12.02已更新）
提取码：59al

第二步：运行genar_train_data.py 其中训练集路径记得修改一下

第三步：开始训练 cifar10_train.py

# 测试效果
cifar10_imagenet_eval.py
使用训练集评估模型效果

cifar10_runsingle.py
单独识别一张图片分类

详细解说文章：https://www.cnblogs.com/7rhythm/p/7091624.html  源代码与文章中代码片段已经有所不同，注意区分

# license
Apache License

https://github.com/tensorflow/tensorflow/blob/master/LICENSE



# -------en---------

# MYcifar255 profile
1. Improved according to the tensorflow-cifar10 example to accommodate more pictures and classifications.2. Improve the generation mechanism of packing pictures into Bin files

# Deployment
Python 3.5
Numpy = = 1.12.0
Tensorflow = = 1.2.1

# Usage steps 

Step 1: download the training data set (more than 200 image categories, image source: ILSVRC2012, image compressed)
https://pan.baidu.com/s/1bDgwtiwNjYOHNCoBOXcEbA extracted code: s1yw
Step 2: run genar_train_data.py where path of the training set is modified
Step 3: run the cifar10_train.py

# Test
Cifar10_imagenet_eval.Py
The effect of the model was evaluated using the training set

Cifar10_runsingle.Py
Identify a single image for classification

explanation article: https://www.cnblogs.com/7rhythm/p/7091624.html （source code and article code snippet is different, pay attention to distinguish）

