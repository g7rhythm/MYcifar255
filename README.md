# MYcifar255 简介
1，根据tensorflow -cifar10 示例 改进，以便适应更多图片与分类。2，完善打包图片到Bin文件的生成机制

# Deployment
python 3.5

numpy==1.12.0

tensorflow==1.2.1

# 使用步骤
第一步：下载训练数据集  （200多种图片分类，图片来源：ILSVRC2012，图片压缩过）
https://pan.baidu.com/s/1bDgwtiwNjYOHNCoBOXcEbA      提取码：s1yw

第二步：运行genar_train_data.py 其中训练集路径记得修改一下

第三步：开始训练 cifar10_train.py

# 测试效果
cifar10_imagenet_eval.py
使用训练集评估模型效果

cifar10_runsingle.py
单独识别一张图片分类


# license
Apache License

https://github.com/tensorflow/tensorflow/blob/master/LICENSE

