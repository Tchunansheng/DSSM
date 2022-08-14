from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import sys
import collections
import os
import time
import random
from sklearn.cluster import DBSCAN

project = 'DSSM'
sys.path.append(os.getcwd().split(project)[0]+project)
print(os.getcwd().split(project)[0]+project)

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dssm import datasets
from dssm.trainers import Trainer
from dssm.evaluators import Evaluator, extract_features
from dssm.utils.data import IterLoader
from dssm.utils.data import transforms as T
from dssm.utils.data.sampler import RandomMultipleGallerySampler
from dssm.utils.data.preprocessor import Preprocessor
from dssm.utils.logging import Logger
from dssm.utils.serialization import load_checkpoint, save_checkpoint
from dssm.utils.rerank import compute_jaccard_dist
from dssm.models.resnet import Model, convert_dsbn

start_epoch = best_mAP = 0

def str2bool(v):
    if isinstance(v,bool):
        return v
    if v == 'True':
        return True
    if v == 'False':
        return False

def get_data(name, data_dir):
    root = osp.join(data_dir,name)
    dataset = datasets.create(name, root)
    return dataset

def get_train_loader(dataset, height, width, batch_size, workers,
                    num_instances, iters, trainset=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
	         T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
         ])

    train_set = sorted(dataset.train) if trainset is None else trainset
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
                DataLoader(Preprocessor(train_set, root=dataset.images_dir,
                                        transform=train_transformer, mutual=False),
                            batch_size=batch_size, num_workers=workers, sampler=sampler,
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)
    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])
    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    return test_loader

def create_model(source_classes,target_images):
    model = Model(source_classes, target_images)
    return model

# generate sampling rate randomly
def gaussian_rate(mean,std,low_peak,high_peak):
    import numpy as np
    while True:
        rate = np.random.normal(mean,std,1)
        rate = np.float(rate[0])
        if low_peak<rate<high_peak:
            return rate

# calculate R-MHS weight for each sample
def cal_weight_from_sample(features,labels,temp=1.0):
    dis=[]
    features = F.normalize(features,dim=1)
    similary = features.matmul(features.t())
    sim_bottom, sim_index = torch.sort(similary,descending=True)
    for i in range(features.size(0)):
        if labels[i]==-1:
            dis.append(100)
            continue
        for sort_index,sim in zip(sim_index[i],sim_bottom[i]):
            if (labels[i] != labels[sort_index]) and (labels[sort_index]!=-1):
                dis.append(sim)
                break
    dis = torch.tensor(dis)
    dis = -dis/temp
    dis = torch.softmax(dis,dim=0)*features.size(0)
    return dis


def main():
    args = parser.parse_args()
    main_worker(args)

def main_worker(args):
    global start_epoch, best_mAP
    cudnn.benchmark = True
    print(torch.cuda.device_count())
    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    iters = args.iters if (args.iters>0) else None
    dataset_source = get_data(args.dataset_source, args.data_dir)
    dataset_target = get_data(args.dataset_target, args.data_dir)
    test_loader_target = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers)
    train_loader_target = get_train_loader(dataset_target, args.height, args.width,args.batch_size, args.workers, args.num_instances, iters)
    train_loader_source = get_train_loader(dataset_source, args.height, args.width,args.batch_size, args.workers, args.num_instances, iters)

    source_pid_num = dataset_source.num_train_pids
    target_img_num = dataset_target.num_train_imgs
    print('train dataset imgs number', target_img_num)
    # Create model
    model = create_model(source_pid_num, 2500)
    model.cuda()
    model = nn.DataParallel(model)

    model_ema = create_model(source_pid_num, 2500)
    model_ema.cuda()
    model_ema = nn.DataParallel(model_ema)

    evaluator_ema = Evaluator(model_ema)

    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]

    for para in model_ema.parameters():
        para.requires_grad = False

    optimizer = torch.optim.Adam(params)
    trainer = Trainer(model=model, model_ema=model_ema,pretrain_epoch=args.pretrain_epoch,src_classes=source_pid_num,beta=args.beta)
    args.num_clusters=2500

    for epoch in range(start_epoch,args.epochs):
        start = time.time()

        if epoch == args.pretrain_epoch:
            model = create_model(source_pid_num, 2500)
            model_ema = create_model(source_pid_num, 2500)
            checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_checkpoint.pth.tar'))['state_dict']
            new_pre={}
            for k,v in checkpoint.items():
                name=k[7:]
                new_pre[name]=v

            model.load_state_dict(new_pre)
            convert_dsbn(model)
            model.cuda()
            model = nn.DataParallel(model)

            model_ema.load_state_dict(new_pre)
            convert_dsbn(model_ema)
            model_ema.cuda()
            model_ema = nn.DataParallel(model_ema)

            evaluator_ema = Evaluator(model_ema)
            params = []
            for key, value in model.named_parameters():
                if not value.requires_grad:
                    continue
                params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
            for para in model_ema.parameters():
                para.requires_grad = False

            optimizer = torch.optim.Adam(params)
            trainer = Trainer(model=model, model_ema=model_ema,pretrain_epoch=args.pretrain_epoch,src_classes=source_pid_num,beta=args.beta)
            del checkpoint, new_pre

        if epoch > (args.pretrain_epoch-1):
            if args.rate_function =='gaussian':
                rate = gaussian_rate(args.mean,args.std,args.low_peak,args.high_peak)
            if args.rate_function == 'stable':
                rate = args.mean

            random_num = int(target_img_num * rate)
            tgt_cluster_dataset = random.sample(dataset_target.train, random_num)
            print('sampling rate:',rate)
            tgt_cluster_loader = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers, testset=tgt_cluster_dataset)

            # extract features for clustering
            _, dict_f, _ = extract_features(model_ema, tgt_cluster_loader, print_freq=100)
            cf = torch.stack(list(dict_f.values()))
            rerank_dist = compute_jaccard_dist(cf, use_gpu=False).numpy()
            cluster = DBSCAN(eps=args.eps, min_samples=args.min_samples, metric='precomputed', n_jobs=-1)
            print('Clustering and labeling...')
            labels = cluster.fit_predict(rerank_dist)
            num_ids = len(set(labels)) - (1 if -1 in labels else 0)
            args.num_clusters = num_ids
            print('** Clustered into {} classes '.format(args.num_clusters))

            weights = cal_weight_from_sample(cf,labels,args.temp)

            # extrcat features for initializing target domain's classifier
            _, dict_f, _ = extract_features(model, tgt_cluster_loader, print_freq=100)
            cf = torch.stack(list(dict_f.values()))

            new_dataset = []
            cluster_centers = collections.defaultdict(list)
            if args.with_rmhs:
                for index,((fname,_,camid,_),label,tgt_weight) in enumerate(zip(tgt_cluster_dataset,labels,weights)):
                    if label == -1: continue
                    new_dataset.append((fname, label, camid, tgt_weight))
                    cluster_centers[label].append(cf[index])
            else:
                for index,((fname,_,camid,_),label) in enumerate(zip(tgt_cluster_dataset,labels)):
                    if label == -1: continue
                    new_dataset.append((fname, label, camid, 1.0))
                    cluster_centers[label].append(cf[index])

            print('new dataset length ', len(new_dataset))
            cluster_centers = [torch.stack(cluster_centers[idx]).mean(0) for idx in sorted(cluster_centers.keys())]
            cluster_centers = torch.stack(cluster_centers)
            cluster_centers = F.normalize(cluster_centers, dim=1)

            model.module.classifier.weight.data[source_pid_num:args.num_clusters+source_pid_num].copy_(cluster_centers.float().cuda())

            train_loader_target = get_train_loader(dataset_target, args.height, args.width,
                                                args.batch_size, args.workers, args.num_instances, iters, trainset=new_dataset)
            del dict_f, cf, rerank_dist, cluster_centers

        train_loader_source.new_epoch()
        train_loader_target.new_epoch()
        trainer.train(epoch=epoch, data_loader_src=train_loader_source, data_loader_target=train_loader_target,
                      optimizer=optimizer, train_iters=len(train_loader_target), num_cluster=args.num_clusters)

        def save_model(model, fname,cluster_num=0):
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch ,
                'cluster_num':cluster_num,
            },  fpath=fname)

        if (epoch == args.pretrain_epoch-1):
            save_model(model,osp.join(args.logs_dir, 'model_checkpoint.pth.tar'))

        if epoch % 5 == 0 or epoch==args.epochs-1):
            print('test on target:')
            _,mAP = evaluator_ema.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True)
            best_mAP = max(mAP, best_mAP)
            print('Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}'.
                  format(epoch, mAP, best_mAP))

        end = time.time()
        time_elapsed = end - start
        print('Training epoch: {} complete in {:.0f}m {:.0f}\n'.format(epoch,time_elapsed // 60, time_elapsed % 60))

    print ('finished!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MMT Training")
    # market1501  msmt17_v1   dukemtmc
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--num_instances', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.00035)
    parser.add_argument('--alpha', type=float, default=0.999)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--beta', type=float, default=0.999)
    parser.add_argument('--min_samples', type=int, default=4)
    parser.add_argument('--iters', type=int, default=500)
    parser.add_argument('--pretrain_epoch', type=int, default=1)
    parser.add_argument('--data_dir', type=str, metavar='PATH', default='/root/data/unzip/')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--logs_dir', type=str, metavar='PATH', default='../model_path/try6')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--temp', type=float, default=0.3)

    parser.add_argument('--dataset_source', type=str, default='dukemtmc')
    parser.add_argument('--dataset_target', type=str, default='market')
    parser.add_argument('--eps', type=float, default=0.56)
    parser.add_argument('--low_peak', type=float, default=0.3)
    parser.add_argument('--high_peak', type=float, default=1.0)

    parser.add_argument('--rate_function', type=str, default='gaussian')# gaussian for DSSM; stable for SSSM
    parser.add_argument('--mean', type=float, default=0.6)
    parser.add_argument('--std', type=float, default=0.2)
    parser.add_argument('--with_rmhs', type=str2bool, default=False)

    main()
