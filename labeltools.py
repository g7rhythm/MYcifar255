import os

import tensorflow as tf
def diff(re2,key):
    keys = tf.fill([tf.size(re2)],key[0])
    numpoi= tf.cast(tf.equal(re2, keys),tf.int32)
    numpoi=tf.argmax(numpoi)
    return numpoi
def splitfilenames(inputs,allstringlen):
    a = tf.string_split(inputs, "/\\")
    bigin = tf.cast(tf.size(a.values) / allstringlen -2, tf.int32)
    slitsinglelen = tf.cast(tf.size(a.values) / allstringlen, tf.int32)
    val = tf.reshape(a.values, [allstringlen, slitsinglelen])
    re2 = tf.cast(tf.slice(val, [0, bigin], [allstringlen, 1]),tf.string)
    re2 =tf.reshape(re2,[allstringlen])
    re2 =tf.unique(re2).y
    return re2
def getfilelist(data_dir):
    filenames=[]
    categorylist = sorted(os.listdir(data_dir))
    i = 1
    for item in categorylist:
        print("load file type:" + str(i))
        fpath = ''.join([data_dir,'\\',item])
        tmp = os.listdir(fpath)
        for seitem in tmp:
            fulseitem = [fpath,'\\', seitem]
            filenames.append(''.join(fulseitem).replace("\\",'/'))
        i += 1
    return filenames,categorylist

def PackageMultImageIntoBin(files,allFilesDir,targetpath):
    if tf.gfile.Exists(targetpath):
        tf.gfile.DeleteRecursively(targetpath)
    else:
        tf.gfile.MakeDirs(targetpath)
    from PIL import Image
    import numpy as np
    maxLabelCount=255
    categorylist = sorted(os.listdir(allFilesDir))
    i=1
    c=0
    resetW =32
    resetH =32
    batch=list([])
    categoryDic=GetDic(categorylist)
    if len(categoryDic) > maxLabelCount:
      print("label只占1位unit8 最大不能超过255")
      return
    print("shuffle files start")
    files=shuffle(files)
    for imgsrc in files:
      im = Image.open(imgsrc)
      im = im.resize((resetW, resetH))
      im = np.array(im,np.uint8)
      r = im[:, :, 0].flatten()
      g = im[:, :, 1].flatten()
      b = im[:, :, 2].flatten()
      fileByloneLabel=str(imgsrc).split("/")[-2]
      find=categoryDic.get(fileByloneLabel,-1)
      if find ==-1:
          continue
      label=[find]
      temp=list(label) + list(r) + list(g) + list(b)
      if len(temp)!=3*32*32+1:
          print("打包数组总数不能超过原总数")
          return
      batch+=temp

      if(i%2000==0):
          out = np.array(batch, np.uint8)
          c+=1
          out.tofile(targetpath + "/eval_batch_"+str(c)+".bin")
          print("out:%s" % (len(out)))
          batch = []
      i+=1

def shuffle(lis):
    import  random
    result = lis[:]
    for i in range(1, len(lis)):
        j = random.randrange(0, i)
        result[i] = result[j]
        result[j] = lis[i]
    return result

def loadOne():
    from PIL import Image
    import numpy as np
    imgsrc="/imagenet/srsdone/evalimg1/n01440764/n01440764_15560.jpg"
    im = Image.open(imgsrc)
    print(im.size)
    im= im.resize((32,32))
    im = (np.array(im))
    temp=im[:, :, 0]
    r = temp.flatten()
    return  r
def GetDic(categorylist):
    dic={}
    i=0
    for item in categorylist:
        i=i+1
        dic[item] = i
    return dic


