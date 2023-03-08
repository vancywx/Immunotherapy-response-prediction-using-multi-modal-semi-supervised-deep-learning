import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
from core.evaluate import validate
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from misc import log
from nets import multiModal_densenet as network
from dataloaders.medicalData_semi import multi_modal
from misc import readfilelist
import utils.util as util
from utils import ramps, losses

# '/home/xiwang/Mount/Data3/'
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='ROIMasked',
                    help='ROIMasked/ROIRaw/Whole')

parser.add_argument('--nb', type=str, default='no',
                    help='no/yes')

parser.add_argument('--num_classes', type=int, default=2,
                    help='number of classes')

parser.add_argument('--root_dataPath', type=str,
                    default='/home/xiwang/DATA/Yuming/Immunotherapy_png/', help='root path of dataset')
parser.add_argument('--root_dataPath_u', type=str,
                    default='/home/xiwang/DATA/Yuming/Others_png/', help='root path of unlabeled dataset')

parser.add_argument('--root_modelPath', type=str,
                    default='/home/xiwang/Models/Yuming_v2/Three_semi_External_9', help='root path of saved model')
parser.add_argument('--net', type=str,
                    default='densenet121', help='network_name')
#configuration of semi-supervised learning
parser.add_argument('--conf_thre', type=float,  default=0.7,
                    help='confidence threshold')

parser.add_argument('--threshold', type=float,  default=0.9,
                    help='threshold')

parser.add_argument('--beta', type=float,  default=1,
                    help='loss weight')


parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=40.0, help='consistency_rampup')
parser.add_argument('--initial_lr', default=0.0, type=float,
                    metavar='LR', help='initial learning rate when using linear rampup')
parser.add_argument('--lr_rampup', default=0, type=int, metavar='EPOCHS',
                    help='length of learning rate rampup in the beginning')
parser.add_argument('--lr_rampdown_epochs', default=None, type=int, metavar='EPOCHS',
                    help='length of learning rate cosine rampdown (>= length of training)')


parser.add_argument('--image_height', type=int, default=160,
                    help='image height')
parser.add_argument('--image_width', type=int, default=160,
                    help='image width')
parser.add_argument('--image_mean', type=int, default=15.74,
                    help='image mean')
parser.add_argument('--image_std', type=int, default=31.46,
                    help='image std')

parser.add_argument('--max_iterations', type=int,
                    default=500000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=12,
                    help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.00001,
                    help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
args = parser.parse_args()

if not os.path.exists(args.root_modelPath):
    os.makedirs(args.root_modelPath)

snapshot_path = "%s/%s_%s_neighbor-%s_class%d_seed%d/" %(args.root_modelPath, args.net, args.mode, args.nb,args.num_classes,args.seed)

print('save models to %s' %(snapshot_path))
args.snapshot_path=snapshot_path
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
print('Batch size: %d' % (batch_size))
max_iterations = args.max_iterations
base_lr = args.base_lr


def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch):
    lr = args.base_lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from
    # https://arxiv.org/abs/1706.02677
    lr = ramps.linear_rampup(epoch, args.lr_rampup) * \
        (args.base_lr - args.initial_lr) + args.initial_lr

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle
    # only)
    if args.lr_rampdown_epochs:
        assert args.lr_rampdown_epochs >= args.epochs
        lr *= ramps.cosine_rampdown(epoch, args.lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def create_model(ema=False):
        # Network definition
    net = network.densenet121(num_classes=args.num_classes)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model




if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
patch_size = (args.image_height, args.image_width, 3)
num_classes = 2


def getDataList(filename):
    with open(filename) as fid:
        lines=fid.readlines()
        data_list=[line.split('\n')[0] for line in lines[1:]]
    return data_list

if __name__ == "__main__":
    # make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    
    
    print('create student model.................................................')
    net = create_model()
    print('student models is created..........................................')
    print('create teacher model.................................................')
    ema_model = create_model(ema=True)
    print('student models is created..........................................')

    device = torch.device('cuda:' + str(args.gpu))
    net = net.to(device)

    Dir = args.root_dataPath
    trainlist, vallist, testlist = getDataList('CSV_files/train.csv'),getDataList('CSV_files/val.csv'),getDataList('CSV_files/External.csv')
    train_nb=getDataList('CSV_files/train_neighbor.csv')
    unlabeledlist = getDataList('CSV_files/unlabeled_shuffled.csv')
    if args.nb=='yes':
        print('Train with neighbors(%d raw images\t%d neighbors)...' %(len(trainlist),len(train_nb)))
        trainlist=trainlist+train_nb
    print ('Train list:', (len(trainlist)))
    print ('Test list:', (len(vallist)))
    print ('External Test list:', (len(testlist)))
    print ('Unlabeled list:', (len(unlabeledlist)))
    trainlist=trainlist * int(len(unlabeledlist)/len(trainlist))
    trainlist_real=getDataList('CSV_files/train.csv')
    
    if args.mode=='ROIMasked':
        dataset_mean_value = 15.74 / 255.0
        dataset_std_value = 31.46 / 255.0
    elif args.mode=='ROIRaw':
        dataset_mean_value = 32.28 / 255.0
        dataset_std_value = 37.12 / 255.0
    elif args.mode=='Whole':
        dataset_mean_value = 37.521 / 255.0
        dataset_std_value = 48.836 / 255.0
        
        
    print('Mode: %s;image mean: %.4f; image std: %.4f' %(args.mode, dataset_mean_value,dataset_std_value))


    dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
    dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)
    data_root=args.root_dataPath
    data_root_u=args.root_dataPath_u
    
    db_train = multi_modal(data_root, trainlist, patch_size, args.mode, args.num_classes,
                        transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.ToTensor(),
                                            transforms.Normalize(
                            mean=dataset_mean,
                            std=dataset_std)]),
                        transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=dataset_mean,
                                std=dataset_std)]))
    db_train_u = multi_modal(data_root_u, unlabeledlist, patch_size, args.mode, args.num_classes,
                        transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.ToTensor(),
                                            transforms.Normalize(
                            mean=dataset_mean,
                            std=dataset_std)]),
                        transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=dataset_mean,
                                std=dataset_std)]))
    db_train_real = multi_modal(data_root, trainlist_real, patch_size, args.mode,args.num_classes,
                       transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize(
                                               mean=dataset_mean,
                                          std=dataset_std)]),
                       transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=dataset_mean,
                                std=dataset_std)]))
    db_test = multi_modal(data_root, vallist, patch_size, args.mode,args.num_classes,
                       transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize(
                                               mean=dataset_mean,
                                          std=dataset_std)]),
                       transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=dataset_mean,
                                std=dataset_std)]))
    db_externalTest = multi_modal(data_root, testlist, patch_size, args.mode,args.num_classes,
                       transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize(
                                               mean=dataset_mean,
                                          std=dataset_std)]),
                       transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=dataset_mean,
                                std=dataset_std)]))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)
    trainReal_loader = DataLoader(db_train_real, batch_size=batch_size * 2, shuffle=False,
                            num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)
    testloader = DataLoader(db_test, batch_size=batch_size * 2, shuffle=False,
                            num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)
    external_testloader = DataLoader(db_externalTest, batch_size=batch_size * 2, shuffle=False,
                            num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)
    trainloader_u = DataLoader(db_train_u, batch_size=batch_size, shuffle=True,
                               num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)

    net.train()
    ema_model.train()
#     optimizer = optim.SGD(net.parameters(), lr=base_lr,
#                           momentum=0.9, weight_decay=0.0001)
    optimizer = optim.Adam(net.parameters(), lr=base_lr, weight_decay=0.0001)
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    print('max_epoch',max_epoch)
    max_epoch=8
    lr_ = base_lr
    net.train()
    ema_model.train()

    w_id = open('%s/TrainingRecord.txt' % (snapshot_path), 'w')

    w_id.write('Initial lr=%f\n' % (base_lr))
    w_id.close()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        args.epoch_num=epoch_num
        net.train()
        ema_model.train()
        time1 = time.time()
        ce_losses = util.AverageMeter()
        ce_accs = util.AverageMeter()
        consist_losses=util.AverageMeter()
        fm_losses = util.AverageMeter()
        unlabel_iter = iter(trainloader_u)
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            image_batch, clinic_batch, radio_batch, label_batch = sampled_batch['image'],sampled_batch['clinic'],sampled_batch['radio'], sampled_batch['label']
            image_batch, clinic_batch, radio_batch, label_batch = image_batch.to(device), clinic_batch.to(device), radio_batch.to(device),label_batch.to(device)
            
            outputs = net([image_batch,clinic_batch,radio_batch])
            outputs_soft = F.softmax(outputs, dim=1)
            # (1) supervised loss
            cross_entropy = torch.nn.CrossEntropyLoss()
            loss_clf = cross_entropy(outputs, label_batch)
            # (2) consistency loss between the student model and the teacher model
            ema_inputs = image_batch
            with torch.no_grad():
                ema_output = ema_model([ema_inputs,clinic_batch,radio_batch] )

            consistency_weight = get_current_consistency_weight(epoch_num)
            consistency_dist = torch.mean(consistency_criterion(
                outputs, ema_output))
            consistency_loss = consistency_weight * consistency_dist
            
            # unlabeled data
            try:
                unlabeled_batch = next(unlabel_iter)
            except StopIteration:
                unlabel_iter = iter(trainloader_u)
                unlabeled_batch = next(unlabel_iter)
            image_batch_u, image_batch_u_s, clinic_batch_u, radio_batch_u = unlabeled_batch['image'], unlabeled_batch['image_s'],unlabeled_batch['clinic'],unlabeled_batch['radio']
            image_batch_u, image_batch_u_s, clinic_batch_u, radio_batch_u = image_batch_u.cuda(), image_batch_u_s.cuda(), clinic_batch_u.cuda(), radio_batch_u.cuda() 

            # Generate artificial label
            with torch.no_grad():
                output_u = F.softmax(ema_model([image_batch_u,clinic_batch_u, radio_batch_u]), 1)
                max_prob, label_u = output_u.max(1)
                mask_u = (max_prob >= args.conf_thre).float()

            # (3) unsupervised loss
            output_u = net([image_batch_u_s,clinic_batch_u, radio_batch_u])
            loss_u = F.cross_entropy(output_u, label_u, reduction='none')
            loss_u = (loss_u * mask_u).mean()

            loss = loss_clf + consistency_loss + 0.001* loss_u
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(net, ema_model, args.ema_decay, iter_num)

            acc = util.accuracy(outputs_soft, label_batch)
            ce_losses.update(loss_clf, image_batch.shape[0])
            consist_losses.update(consistency_loss, image_batch.shape[0])
            fm_losses.update(loss_u, image_batch_u.shape[0])
            ce_accs.update(acc, image_batch.shape[0])

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/clf_loss', loss_clf, iter_num)
            writer.add_scalar('loss/consist_loss', consistency_loss, iter_num)
            writer.add_scalar('loss/fm_loss', loss_u, iter_num)
            logging.info('iteration %d : loss : %f, ce_loss: %f, consistency_loss: %f, fm_loss: %f' % (
                iter_num, loss.item(), loss_clf.item(), consistency_loss.item(), loss_u.item()))

            if iter_num % 50 == 0:
                logging.info('iteration %d : accumulated mean loss : %f; consistency_loss: %f, fm_loss: %f; accumulated mean acc : %f' % (
                    iter_num, ce_losses.avg, consist_losses.avg, fm_losses.avg, ce_accs.avg))
                w_id = open('%s/TrainingRecord.txt' % (snapshot_path), 'a')
                w_id.write('Iteration: %d\t train: CE_Loss: %.4f \t train: CR_Loss: %.4f; train: FM_Loss: %.4f; Accuracy: %.4f\n' % (
                    iter_num, ce_losses.avg, consist_losses.avg, fm_losses.avg, ce_accs.avg))
                w_id.close()

            # change lr
            # if iter_num % 7500 == 0:
            #     lr_ = base_lr * 0.1 ** (iter_num // 7500)
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr_
            if iter_num % 250 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                state = {'epoch': epoch_num + 1,
                         'global_step': iter_num,
                         'state_dict': net.state_dict(),
                         'ema_state_dict': ema_model.state_dict(),
                         'optimizer': optimizer.state_dict(), }
                torch.save(state, save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num > max_iterations:
                break
            time1 = time.time()
        if iter_num > max_iterations:
            break
        # change lr
        if (epoch_num+1) % 3 == 0:
            lr_ = base_lr * 0.1 ** ((epoch_num+1) // 3)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
        steps = int(np.ceil(len(trainlist_real) / (args.batch_size*2)))
        bar = log.ProgressBar(total=steps, width=40)
        print('---------------------------------------------------------------------')
        print('Validation on the training set begins...')
        args.val_mode='train'
        args.val_list=trainlist_real
        acc, sensitivity, specificity, precision,  F1, auc, kappa = validate(
            args, ema_model, trainReal_loader, bar)
        w_id = open('%s/TrainingRecord.txt' % (snapshot_path), 'a')
        w_id.write('Epoch: %d\t(Training set)Accuracy:%.4f\tSensitivity:%.4f\tSpecificity:%.4f\tPrecision:%.4f\tF1:%.4f\tAUC: %.4f\tKappa score:%.4f\n' % (
            epoch_num, acc, sensitivity, specificity, precision, F1, auc, kappa))
        w_id.close()
        print('Validation ends...')
        print('---------------------------------------------------------------------')
        
        del(steps)
        del(bar)
        
        steps = int(np.ceil(len(vallist) / (2.0 * args.batch_size)))
        bar = log.ProgressBar(total=steps, width=40)
        print('---------------------------------------------------------------------')
        print('Validation on the test set begins...')
        args.val_mode='val'
        args.val_list=vallist
        acc, sensitivity, specificity, precision,  F1, auc, kappa = validate(
            args, ema_model, testloader, bar)
        auc1=auc
        w_id = open('%s/TrainingRecord.txt' % (snapshot_path), 'a')
        w_id.write('Epoch: %d\t(Testing set)Accuracy:%.4f\tSensitivity:%.4f\tSpecificity:%.4f\tPrecision:%.4f\tF1:%.4f\tAUC: %.4f\tKappa score:%.4f\n' % (
            epoch_num, acc, sensitivity, specificity, precision, F1, auc, kappa))
        w_id.close()
        print('Validation ends...')
        print('---------------------------------------------------------------------')
        
        del(steps)
        del(bar)
        steps = int(np.ceil(len(testlist) / (2.0 * args.batch_size)))
        bar = log.ProgressBar(total=steps, width=40)
        print('---------------------------------------------------------------------')
        print('Validation on the external test set begins...')
        args.val_mode='test'
        args.val_list=testlist
        acc, sensitivity, specificity, precision,  F1, auc, kappa = validate(
            args, ema_model, external_testloader, bar)
        auc2=auc
        w_id = open('%s/TrainingRecord.txt' % (snapshot_path), 'a')
        w_id.write('Epoch: %d\t(External set)Accuracy:%.4f\tSensitivity:%.4f\tSpecificity:%.4f\tPrecision:%.4f\tF1:%.4f\tAUC: %.4f\tKappa score:%.4f\n' % (
            epoch_num, acc, sensitivity, specificity, precision, F1, auc, kappa))
        w_id.close()
        
        print('Validation ends...')
        print('---------------------------------------------------------------------')
        
        
        save_mode_path = os.path.join(
            snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        state = {'epoch': epoch_num + 1,
             'global_step': iter_num,
             'state_dict': net.state_dict(),
             'ema_state_dict': ema_model.state_dict(),
             'optimizer': optimizer.state_dict(), }
        torch.save(state, save_mode_path)
        logging.info("save model to {}".format(save_mode_path))

    save_mode_path = os.path.join(
        snapshot_path, 'iter_' + str(max_iterations + 1) + '.pth')
    state = {'epoch': epoch_num + 1,
             'global_step': iter_num,
             'state_dict': net.state_dict(),
             'ema_state_dict': ema_model.state_dict(),
             'optimizer': optimizer.state_dict(), }
    torch.save(state, save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
