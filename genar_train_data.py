import labeltools
if __name__ == '__main__':
    alldata_dir = '/imagenet/srsdone/forTrain'
    traindatadir=alldata_dir
    trainTargetpath = '/tmp/cifar10_data'
    #在上面alldata_dir里摘取一些作为测试评估集的数据
    evaldatadir = '/imagenet/srsdone/forEval'
    filenames, _ = labeltools.getfilelist(alldata_dir)
    evalTargetpath = '/tmp/cifar10_eval'
    labeltools.PackageMultImageIntoBin(filenames, alldata_dir, trainTargetpath)
    labeltools.PackageMultImageIntoBin(filenames, evaldatadir, evalTargetpath)
