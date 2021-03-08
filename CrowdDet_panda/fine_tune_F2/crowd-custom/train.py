import os
import argparse
from setproctitle import setproctitle

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

import network
import misc_utils
from config import config
from dataset import train_dataset, eval_dataset, multi_train_dataset

from dataset_map import datasets_map
from logger import Logger
def process(args,data_generater, num_gpus):
    images = []
    gt_boxes = []
    im_info = []
    masks = []
    aug_images = []
    done_an_epoch = False
    for _ in range(num_gpus):
        data = data_generater.__next__()
        minibt_img, minibt_gt, minibt_info = \
                data['data'], data['boxes'], data['im_info']
        images.append(minibt_img)
        gt_boxes.append(minibt_gt)
        im_info.append(minibt_info)
        if args.flip_aug:
            aug_images.append(data['aug_data'])
        if data['done']:
            done_an_epoch = True
    images = torch.Tensor(np.vstack(images)).cuda()
    gt_boxes = torch.Tensor(np.vstack(gt_boxes)).cuda()
    im_info = torch.Tensor(np.vstack(im_info)).cuda()
    extra = {}
    if args.flip_aug:
        extra['aug_data'] = torch.Tensor(np.vstack(aug_images)).cuda()
    return images, gt_boxes, im_info, done_an_epoch, extra

def adjust_learning_rate(optimizer,epoch,lr):
    learning_rate = lr
    lr = learning_rate*(0.1**(epoch//10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train(args):
    if type(config.train_source) == list:
        training_data = multi_train_dataset(args)
    else:
        training_data = train_dataset(args)
    number_of_training_instances = training_data.__next__()
    val_data = eval_dataset(args)
    number_of_val_instances = val_data.__next__()

    total_nr_iters = args.epochs * number_of_training_instances
    batch_per_gpu = config.train_batch_per_gpu
    
    base_lr = config.base_lr
    line = 'network.base_lr.{}.train_iter.{}'.format(base_lr, total_nr_iters)

    print(line)

    # set model save path and log save path
    saveDir = config.model_dir
    misc_utils.ensure_dir(saveDir)

    # set data input pipe
    program_name = config.program_name
    # check gpus
    torch.set_default_tensor_type('torch.FloatTensor')
    if not torch.cuda.is_available():
        print('No GPU exists!')
        return
    else:
        num_gpus = torch.cuda.device_count()

        train_iter = total_nr_iters//(num_gpus*batch_per_gpu)
        
        print('[-]',num_gpus, batch_per_gpu,total_nr_iters)

        new_decay = (np.array(config.lr_decay) / 450000) * total_nr_iters

        train_lr_decay = new_decay //(num_gpus*batch_per_gpu)

        train_dump_interval = number_of_training_instances //(num_gpus*batch_per_gpu)

    train_lr = base_lr * num_gpus
    bt_size = num_gpus * batch_per_gpu

    line = 'Num of GPUs:{}, learning rate:{:.5f}, batch size:{},\
            train_iter:{}, decay_iter:{}, dump_interval:{}'.format(
            num_gpus,train_lr,bt_size,train_iter,train_lr_decay, train_dump_interval)
    print(line)

    print("[-]Building netowrk.")
    net = network.Network(args)
    net.cuda()

    best = 10e10
    epoch = 0
    if args.resume:
        print("Load base model from :",os.path.join(args.save_dir,args.output_name,'dump_last.pth'))
        check_point = torch.load(os.path.join(args.save_dir,args.output_name,'dump_last.pth'))
        net.load_state_dict(check_point['state_dict'])
        start_iter = check_point['step']
        if 'val_loss' in check_point:
            best = check_point['val_loss']
        epoch = start_iter // train_dump_interval + 1 
    elif args.base_model:
        print("Load base model from :",args.base_model)
        check_point = torch.load(args.base_model)
        net.load_state_dict(check_point['state_dict'], strict=False)
        start_iter = 0
    else:
        start_iter = 0
 

    net = nn.DataParallel(net)
    # set the optimizer, use momentum and weight_decay
    optimizer = optim.SGD(net.parameters(), lr=train_lr, momentum=config.momentum, \
        weight_decay=config.weight_decay)

    if(start_iter >= train_lr_decay[0]):
            optimizer.param_groups[0]['lr'] = train_lr / 10
    if(start_iter >= train_lr_decay[1]):
        optimizer.param_groups[0]['lr'] = train_lr / 100   
        
    # check if resume training
    net.train()
    logger = Logger(args)

    iter_tqdm = None
    val_tqdm = None
    for step in range(start_iter, train_iter):
        # warm up
        if step < config.warm_iter:
            alpha = step / config.warm_iter
            lr_new = 0.1 * train_lr + 0.9 * alpha * train_lr
            optimizer.param_groups[0]['lr'] = lr_new
        elif step == config.warm_iter:
            optimizer.param_groups[0]['lr'] = train_lr
        if step == train_lr_decay[0]:
            optimizer.param_groups[0]['lr'] = train_lr / 10
        elif step == train_lr_decay[1]:
            optimizer.param_groups[0]['lr'] = train_lr / 100
        # get training data
        images, gt_boxes, img_info, done_an_epoch, extra = process(args,training_data, num_gpus)
        if done_an_epoch :
            epoch += 1
        optimizer.zero_grad()
        # forward
        outputs = net(images, img_info, gt_boxes,extra = extra)
        # collect the loss
        total_loss = sum([outputs[key].mean() for key in outputs.keys()])
        total_loss.backward()
        optimizer.step()

        # stastic
        stastic_total_loss = total_loss.cpu().data.numpy()
        line = '[*]Epoch:{} iter<{}> lr:{:.5f}, loss:{:.4f}'.format(
            epoch,step, optimizer.param_groups[0]['lr'], float(stastic_total_loss))
            
        
        if step % config.log_dump_interval == 0:
            logger.scalar_summary('lr', optimizer.param_groups[0]['lr'], step)
            for k,v in outputs.items():
                v = float(np.mean(v.cpu().data.numpy()))
                logger.scalar_summary(k, v, step)
                line += ', '+k + ':{:.4}'.format(v)
            logger.scalar_summary('total_loss', float(stastic_total_loss), step)
        else:
            for k,v in outputs.items():
                    v = float(np.mean(v.cpu().data.numpy()))
                    line += ', '+k + ':{:.4}'.format(v)
        if iter_tqdm is None:
            iter_tqdm = tqdm(total=train_iter, desc='Iteration')
            iter_tqdm.update(start_iter)
        iter_tqdm.set_description("[-] "+line)
        iter_tqdm.refresh()
        # save the best model
        if done_an_epoch:
            if args.save_per_epoch > 0:
                if (epoch+1) % args.save_per_epoch == 0:
                    fpath = os.path.join(saveDir,'dump_{}.pth'.format(epoch))
                    print('[.] Saving :',fpath)
                    model = dict(
                        epoch = epoch,
                        step = step,
                        state_dict = net.module.state_dict(),
                        optimizer = optimizer.state_dict())
                    torch.save(model,fpath)


            fpath = os.path.join(saveDir,'dump_last.pth')
            print('[.] Saving :',fpath)
            model = dict(
                epoch = epoch,
                step = step,
                state_dict = net.module.state_dict(),
                optimizer = optimizer.state_dict())
            torch.save(model,fpath)


        net.train()

        iter_tqdm.update(1)
    iter_tqdm.close()
    
    fpath = os.path.join(saveDir,'dump_last.pth')
    print('[.] Saving :',fpath)
    model = dict(step = step,
        state_dict = net.module.state_dict(),
        optimizer = optimizer.state_dict())
    torch.save(model,fpath)

def updateConfig_train(args):
    dataset_paths = datasets_map[args.dataset]
    config.image_folder = dataset_paths['img_path']
    config.train_source = dataset_paths['json_path']
    config.eval_source = dataset_paths['val_json_path']
    print(dataset_paths)
    config.model_dir = os.path.join(args.save_dir,args.output_name)
    
    config.init_weights = args.init_weights
    #config.output_dir = os.path.join(args.output_dir,args.output_name)
    config.log_dump_interval = args.log_dump_iter
    config.train_batch_per_gpu = args.batch_size_per_gpu
    config.base_lr = 1e-5 * config.train_batch_per_gpu * 1.25
    if args.flip_JSD_0g:
        args.flip_JSD = True

    #config.train_base_iters = args.iterations
    if not args.debug_dir:
        args.debug_dir = os.path.join(config.model_dir,'debug')
    train(args)
def main():
    setproctitle('train ' + os.path.split(os.path.realpath(__file__))[0])
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_weights', '-r', default=None,type=str,help='a number for certain pth file.')
    
    parser.add_argument('--resume', action='store_true',help='Resume the model from save.')

    parser.add_argument('--base_model','-m', default= None)
    parser.add_argument('--save_dir','-sd', default='./exp' )
    parser.add_argument('--output_name','-on', default='default' )

    parser.add_argument('--log_dump_iter',default=500,type=int)



    parser.add_argument('--batch_size_per_gpu',default=8,type=int)

    parser.add_argument('--debug_dir','-dd', default=None )
    parser.add_argument('--dataset', default=None,type=str)
    parser.add_argument('--init_weights', default='../pths/R-50.pkl',type=str,help='Official code require it for init network.')

    parser.add_argument('--epochs', type =int,default = 8) # iteration = epoch * instance_number / batch_size
    parser.add_argument('--save_per_epoch','-spe', type =int,default = -1)
    parser.add_argument('--flip_JSD', action='store_true',help='Semi-supervised learning enable.')
    parser.add_argument('--flip_JSD_0g', action='store_true',help='Semi-supervised learning enable with zero grad on flipped input.')
    parser.add_argument('--recursive', action='store_true',help='recursive mode enable.')
    parser.add_argument('--bit', action='store_true',help='group_norm ResNet50 enable.')
    parser.add_argument('--resnext50_32x4d', action='store_true',help='ResNeXt50 enable.')
    parser.add_argument('--resnext101_32x8d', action='store_true',help='ResNeXt101 enable.')

    parser.add_argument('--diff_loss', action='store_true',help='Diff_loss learning enable.')
    parser.add_argument('--flip_aug', action='store_true',help='')
    parser.add_argument('--FA_heavy', action='store_true',help='heavy flip_aug')

    args = parser.parse_args()
    updateConfig_train(args)

if __name__ == '__main__':
    main()
