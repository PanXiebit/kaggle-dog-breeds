#coding:utf8
import torch as t
import time
# from config import opt
from torchvision.models.resnet import ResNet, BasicBlock
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
class BasicModule(t.nn.Module):
    '''
    封装了nn.Module,这里主要是提供了save和load两个方法
    '''

    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name=str(type(self))# 默认名字

    def load(self, path):
        '''
        可加载指定路径的模型
        '''
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        '''
        保存模型，默认使用“模型名字+时间”作为文件名
        '''
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), name)
        return name
class Flat(t.nn.Module):
    '''
    把输入reshape成（batch_size,dim_length）
    '''

    def __init__(self):
        super(Flat, self).__init__()
        #self.size = size

    def forward(self, x):
        return x.view(x.size(0), -1)

class Mymodel(ResNet, BasicModule):
    def __init__(self, pretrained=False, **kwargs):
        super(Mymodel, self).__init__(BasicBlock, [3, 4, 6, 3])
        pre_model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
        if pretrained:
            pre_model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))  # resnet34
        fc = t.nn.Linear(in_features=1000, out_features=120)
        for param in pre_model.parameters(): # 预训练模型不需要学习参数
            param.require_grad = False
        self.model = t.nn.Sequential(
            pre_model,
            fc
        )
        self.model_name = str(type(self))  # 默认名字

if __name__ == '__main__':
    model = Mymodel(pretrained=True).model


