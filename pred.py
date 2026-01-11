import os
import time
import argparse
import numpy as np
import torch.autograd
from skimage import io, exposure
from torch.utils.data import DataLoader
#################################
from datasets import RS_ST as RS
from models.Network import CSTMNet as Net

DATA_NAME = 'DATA'


#################################

class PredOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        working_path = os.path.dirname(os.path.abspath(__file__))
        parser.add_argument('--pred_batch_size', required=False, default=1, help='prediction batch size')
        parser.add_argument('--test_dir', required=False, default=r'E:\Datasets\WUSU\test',
                            help='directory to test images')
        parser.add_argument('--pred_dir', required=False, default=r'',
                            help='directory to output masks')
        parser.add_argument('--chkpt_path', required=False,
                            default=r'')
        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        self.parser = parser
        return parser.parse_args()

    def parse(self):
        self.opt = self.gather_options()
        return self.opt


def main():
    begin_time = time.time()
    opt = PredOptions().parse()
    net = Net(3, num_classes=RS.num_classes).cuda()
    net.load_state_dict(torch.load(opt.chkpt_path))
    net.eval()

    test_set = RS.Data_test(opt.test_dir)
    test_loader = DataLoader(test_set, batch_size=opt.pred_batch_size)
    predict(net, test_set, test_loader, opt.pred_dir)
    time_use = time.time() - begin_time
    print('Total time: %.2fs' % time_use)


def predict(net, pred_set, pred_loader, pred_dir):
    pred_all_change = os.path.join(pred_dir, 'temporal')
    if not os.path.exists(pred_all_change): os.makedirs(pred_all_change)

    for vi, data in enumerate(pred_loader):
        imgs_1, imgs_2, imgs_3 = data
        imgs_1 = imgs_1.cuda().float()
        imgs_2 = imgs_2.cuda().float()
        imgs_3 = imgs_3.cuda().float()
        mask_name = pred_set.get_mask_name(vi)
        with torch.no_grad():
            change_temporal   = net(imgs_1, imgs_2, imgs_3)
            all_change = torch.softmax(change_temporal,dim=1).detach()
        all_change = torch.argmax(all_change, dim=1).cpu().squeeze().numpy()
        pred_all_path = os.path.join(pred_all_change, mask_name)
        io.imsave(pred_all_path, RS.Index2Color(all_change))
        print(pred_all_path)

if __name__ == '__main__':
    main()