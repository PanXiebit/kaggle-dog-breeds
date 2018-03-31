#coding: utf-8

from config import opt
import torch as t
from data.dataset import DogBreedData
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchnet import meter
from utils.Visualizer import Visualizer
from models.model import Mymodel


def train(**kwargs):
    opt.parse(kwargs)
    vis = Visualizer(opt.env)

    # step1: configure model
    # model = getattr(models, opt.model)() # opt.model = ResNet34 模块内的文件可以看做是它的属性
    model = Mymodel(pretrained=True)
    model = model.model

    # 直接调用torchvision中的resnet34
    # 预训练里面是包括最后全连接层的，所以直接调用会报错： While copying the parameter named fc.weight, whose dimensions in the model are torch.Size([120, 512])
    # and whose dimensions in the checkpoint are torch.Size([1000, 512]).
    # pre_model = resnet34(pretrained=True)
    # Linear = t.nn.Linear(1000, opt.num_classes)
    # model = t.nn.Sequential(
    #     pre_model,
    #     Linear
    # )


    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()

    # step2: data
    train_data = DogBreedData(opt.train_data_root, train=True)
    val_data = DogBreedData(opt.train_data_root, train=False)
    train_dataloader = DataLoader(train_data, opt.batch_size,
                                  shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.batch_size,
                                shuffle=True, num_workers=opt.num_workers)

    # setp3: loss and optimizer
    loss = t.nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    # step4: meters
    loss_meter = meter.AverageValueMeter() # 一个类，计算平均值，方差的
    # confusion_matrix = meter.ConfusionMeter(120)
    previous_loss = 1e100

    # train
    for epoch in range(opt.max_epoch): # one epoch 表示遍历整个数据集
        loss_meter.reset()
        # confusion_matrix.reset()
        # loss = 0 # 损失值清零，计算每一个epoch的损失值的平均值
        for ii,(data, label) in tqdm(enumerate(train_dataloader)):
            input = Variable(data)
            target = Variable(label.type(t.LongTensor))
            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()

            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            score = model(input)
            # 计算交叉熵损失
            loss = loss(score, target) # loss = t.nn.CrossEntropyLoss(_WeightedLoss)也是Module类，这里是forward()函数
            # 反向传播，梯度下降
            loss.backward()
            optimizer.step()

            # loss update and visualize
            # 两种方法对比以下。。
            loss_meter.add(loss.data[0]) # def value(self): return self.mean, self.std
            # confusion_matrix.add(score.data, target.data)
            # 更新loss
            # loss.data[0] += previous_loss
            # loss_old = loss.data[0]
            # loss_mean = loss_old/float(ii)

            if ii%opt.print_freq == opt.print_freq-1:
                # vis.plot('loss', loss_mean)
                vis.plot('loss', loss_meter.value()[0])

        model.save()

        # validate and visualize
        # val_cm, val_accuracy = val(model, val_dataloader)

        # vis.plot('val_accuracy', val_accuracy)
        # vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
            # epoch=epoch, loss=loss_meter.value()[0], val_cm=str(val_cm.value()),
            # train_cm=str(confusion_matrix.value()), lr=opt.lr))

        # update learning rate
        if loss_meter.value()[0] > previous_loss:
            lr = opt.lr * opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss = loss_meter.value()[0]

def val(model,dataloader):
    '''
    计算模型在验证集上的准确率等信息
    '''
    model.eval()
    # confusion_matrix = meter.ConfusionMeter(120) # 混淆矩阵，在类别比较少的时候看起来效果会比准确率好很多，但120类太多了不适合
    for ii, data in tqdm(enumerate(dataloader)):
        input, label = data
        val_input = Variable(input, volatile=True)
        val_label = Variable(label.type(t.LongTensor), volatile=True)
        if opt.use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()
        score = model(val_input)
        # confusion_matrix.add(score.data.squeeze(), label.type(t.LongTensor))

    model.train()
    # cm_value = confusion_matrix.value()
    # accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    # return confusion_matrix, accuracy

if __name__=="__main__":
    print(__file__, __name__)
    import fire
    fire.Fire()

    ## 测试模型，没问题
    # model = Mymodel(pretrained=True)
    # model = model.model
    # print(model) # 最后一个layer：(1): Linear(in_features=1000, out_features=120, bias=True)
    #
    ## 测试数据集，因为是自己的笔记本电脑，所以尽量用很小的数据集吧。 测试没问题～
    # train_data = DogBreedData(opt.train_data_root, train=True)
    # train_dataloader = DataLoader(train_data, opt.batch_size,
    #                               shuffle=True, num_workers=opt.num_workers)
    # for ii, (data, label) in tqdm(enumerate(train_dataloader)): # 通过迭代获取数据
    #     input = Variable(data)
    #     target = Variable(label.type(t.LongTensor))
    #     print(input.size())
    #     print(target.size())



