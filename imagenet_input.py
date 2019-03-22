# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
import labeltools
import cifar10_input
IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 255
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 500
trainPath ='/imagenet/srsdone/forTrain'

#True 读取打包后的二进制文件作为训练数据
#Flase 读取一个个的图文文件作为训练数据 （性能差，仅供学习参考）
GET_TrainFile_By_BIN =True


def read_from_imagenet(filename_queue,filenames):
    class ImageNetRecord(object):
        pass

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)

    image0 = tf.image.decode_jpeg(value,3)

    esized_image = tf.image.resize_images(image0, [32, 32],
                                          method=tf.image.ResizeMethod.AREA)
    result = ImageNetRecord()
    re2=labeltools.splitfilenames(tf.constant( filenames),len(filenames))
    key=labeltools.splitfilenames(tf.reshape(key,[1],name="key_debug"),1)
    label=labeltools.diff(re2,key)
    tf.summary.scalar("label_sum", label)
    result.height = 32
    result.width = 32
    result.depth = 3
    result.label = tf.cast(
        label, tf.int32,name="label_debug")
    result.label = tf.reshape(
        result.label, [1])
    result.uint8image=esized_image
    return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.

    Args:
      image: 3-D Tensor of [height, width, 3] of type.float32.
      label: 1-D Tensor of type.int32
      min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
      batch_size: Number of images per batch.
      shuffle: boolean indicating whether to use a shuffling queue.

    Returns:
      images: Images. 4D tensor of [batch_size, height, width, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.summary.image('images', images)

    return images, tf.reshape(label_batch, [batch_size])

labelDic={}
labels={}


def inputs(eval_data, data_dir, batch_size):
    """Construct input for CIFAR evaluation using the Reader ops.

    Args:
      eval_data: bool, indicating if one should use the train or eval data set.
      data_dir: Path to the CIFAR-10 data directory.
      batch_size: Number of images per batch.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    import  os
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    filenames = []
    if GET_TrainFile_By_BIN:
        tempfilelist = os.listdir("/tmp/cifar10_data")
        for item in tempfilelist:
            if item.find(".bin") != -1:
                filenames.append(os.path.join("/tmp/cifar10_data", item).replace("\\", "/"))
        filename_queue = tf.train.string_input_producer(filenames, name="filename_queue_hcq", shuffle=True)
        read_input = cifar10_input.read_cifar10(filename_queue)

    else:
        data_dir = trainPath
        if len(os.listdir(data_dir))!=NUM_CLASSES:
            raise Exception('图片分类总数与设置NUM_CLASSES参数不一致')
            return
        filenames,calist=labeltools.getfilelist(data_dir)

        filename_queue = tf.train.string_input_producer(filenames, name="filename_queue_hcq", shuffle=True)
        read_input = read_from_imagenet(filename_queue, filenames)

    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           width, height)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(resized_image)

    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])
    #read_input.label.set_shape([])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=False)
