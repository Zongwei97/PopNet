import os
import torch
import torch.nn.functional as F
import sys
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from Code.lib.model import PopNet
from Code.utils.data import get_loader, test_dataset
from Code.utils.utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from Code.utils.options import opt
from loss import *
from Code.lib.sobel import *


get_gradient = Sobel().cuda()
cos = nn.CosineSimilarity(dim=1, eps=0)

# set the device for training
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

cudnn.benchmark = True  # 可以增加程序的运行效率

# build the model
model = PopNet(32, 50)  ### 
if (opt.load is not None):
    model.load_state_dict(torch.load(opt.load))  
    print('load model from ', opt.load)

model.cuda()  
params = model.parameters()  
optimizer = torch.optim.Adam(params, opt.lr)  

# set the path
train_image_root = opt.rgb_label_root
train_gt_root = opt.gt_label_root
train_depth_root = opt.depth_label_root

val_image_root = opt.val_rgb_root
val_gt_root = opt.val_gt_root
val_depth_root = opt.val_depth_root
save_path = opt.save_path

if not os.path.exists(save_path):
    os.makedirs(save_path)  

# load data
print('load data...')
train_loader = get_loader(train_image_root, train_gt_root, train_depth_root, batchsize=opt.batchsize,
                          trainsize=opt.trainsize)
test_loader = test_dataset(val_image_root, val_gt_root, val_depth_root, opt.trainsize)
total_step = len(train_loader)  

logging.basicConfig(filename=save_path + 'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("BBSNet_unif-Train")
logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path,
        opt.decay_epoch))
# set loss function
CE = torch.nn.BCEWithLogitsLoss()
l1ss_loss = SSIM().cuda()

step = 0
writer = SummaryWriter(save_path + 'summary')  #
best_mae = 1  
best_epoch = 0 


    
print(len(train_loader))


def train(train_loader, model, optimizer, epoch, save_path):  
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts, depths) in enumerate(train_loader, start=1):
            optimizer.zero_grad() 
            images = images.cuda() 
            gts = gts.cuda()  
            depths = depths.cuda()  #

            ##
            pre_res = model(images, depths)  

            loss1 = structure_loss(pre_res[0], gts)
            loss2 = structure_loss(pre_res[1], gts)
            loss3 = structure_loss(pre_res[2], gts)


            pre_depth = pre_res[-1]

            pre_depth = (pre_depth - pre_depth.min()) / (pre_depth.max() - pre_depth.min() + 1e-8)
            loss4 = l1ss_loss(pre_depth, depths)
            
            loss5 = smooth_normal_loss(pre_depth*gts)
            
            loss6 = weighted_total_variant(pre_depth,gts) 


            t = pre_res[-2]
            t = (t - t.min()) / (t.max() - t.min() + 1e-8)

            # diff = t - pre_depth (it would be more reasonable using t-pre_depth)
            diff = pre_depth -t
            thresh = F.sigmoid(10*diff)

            
            loss7 = structure_loss(thresh, gts)

            loss = loss1 + loss2 + loss3 + loss4 + loss5 + 0.1*loss6 + 0.1*loss7
            
            loss.backward()  

            clip_gradient(optimizer, opt.clip)  
            optimizer.step()  
            step += 1
            epoch_step += 1
            loss_all += loss.data
            if i % 50 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f} Loss2: {:0.4f} Loss3: {:0.4f} '.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss1.data, loss2.data, loss3.data))
                logging.info(
                    '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f} Loss2: {:0.4f} Loss3: {:0.4f}'.
                    format(epoch, opt.epoch, i, total_step, loss1.data, loss2.data, loss3.data))  # 打印到日志文件中的信息

        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)

        if (epoch) % 5 == 0:
            torch.save(model.state_dict(), save_path + 'HyperNet_epoch_{}.pth'.format(epoch))  # 保留参数模型

    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'HyperNet_epoch_{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise


# test function
def val(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, depth, name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.cuda()
            pre_res = model(image, depth)
            res = pre_res[2]
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'SPNet_epoch_best.pth')
                print('best epoch:{}'.format(epoch))

        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    print("Start train...")

    for epoch in range(1, opt.epoch):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        # train
        train(train_loader, model, optimizer, epoch, save_path)

        # test
        val(test_loader, model, epoch, save_path)
