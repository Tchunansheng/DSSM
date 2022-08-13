from __future__ import print_function, absolute_import
import torch
from .loss import TripletLoss, CrossEntropyLabelSmooth
from .utils.meters import AverageMeter

class Trainer(object):
    def __init__(self, model,model_ema,pretrain_epoch=1,src_classes=None,beta=0.999):
        super(Trainer, self).__init__()
        self.model = model
        self.model_ema = model_ema
        self.pretrain_epoch = pretrain_epoch
        self.src_classes = src_classes
        self.beta=beta
        self.criterion_ce = CrossEntropyLabelSmooth(500).cuda()
        self.hard_tri = TripletLoss(margin=0.3).cuda()

    def train(self, epoch,data_loader_src, data_loader_target,
            optimizer, train_iters=500,num_cluster=500):
        self.model.train()
        self.model_ema.train()

        loss_ce_src_meter = AverageMeter()
        loss_tri_src_meter = AverageMeter()
        loss_ce_tgt_meter = AverageMeter()
        loss_tri_tgt_meter = AverageMeter()

        for i in range(train_iters):
            src_inputs = data_loader_src.next()
            target_inputs = data_loader_target.next()

            src_imgs, src_labels,weight_src,src_fnames = self._parse_data(src_inputs)
            tgt_imgs, tgt_labels,weight_tgt,tgt_fnames = self._parse_data(target_inputs)

            if epoch < self.pretrain_epoch:
                src_prec, src_feat_bn,src_feat_af = self.model(src_imgs)
                _, _, _ = self.model(tgt_imgs)

                loss_src_ce = self.criterion_ce(src_prec[:, :self.src_classes], src_labels, weight_src)
                loss_ce_src_meter.update(loss_src_ce.item())
                loss_src_tri = self.hard_tri(src_feat_bn, src_labels,weight_src)
                loss_tri_src_meter.update(loss_src_tri.item())

                loss_src = loss_src_ce + loss_src_tri
                optimizer.zero_grad()
                loss_src.backward()
                optimizer.step()
                optimizer.zero_grad()

                self._update_ema_variables(self.model,self.model_ema,self.beta,epoch * len(data_loader_target) + i)
                del src_prec, src_feat_bn, loss_src_ce, loss_src_tri,src_feat_af,_

            else:
                device_num = torch.cuda.device_count()
                B,C,H,W = src_imgs.size()
                def reshape(inputs):
                    return inputs.view(device_num, -1, C, H, W)
                s_inputs, t_inputs = reshape(src_imgs), reshape(tgt_imgs)
                inputs = torch.cat((s_inputs, t_inputs),1).view(-1,C,H,W)

                pres, feats_bn, feats_af = self.model(inputs)
                pres_ema, feats_bn_ema, feats_af_ema = self.model_ema(inputs)

                pres = pres.view(device_num, -1, pres.size(-1))
                src_prec, tgt_prec = pres.split(pres.size(1)//2, dim=1)
                src_prec, tgt_prec = src_prec.contiguous().view(-1, pres.size(-1)), tgt_prec.contiguous().view(-1,pres.size(-1))

                feats_bn = feats_bn.view(device_num, -1, feats_bn.size(-1))
                src_feat_bn, tgt_feat_bn = feats_bn.split(feats_bn.size(1)//2,dim=1)
                src_feat_bn, tgt_feat_bn = src_feat_bn.contiguous().view(-1, feats_bn.size(-1)), tgt_feat_bn.contiguous().view(-1,feats_bn.size(-1))

                feats_bn_ema = feats_bn_ema.view(device_num, -1, feats_bn_ema.size(-1))
                src_feat_bn_ema, tgt_feat_bn_ema = feats_bn_ema.split(feats_bn_ema.size(1) // 2, dim=1)
                src_feat_bn_ema, tgt_feat_bn_ema = src_feat_bn_ema.contiguous().view(-1, feats_bn_ema.size(-1)), tgt_feat_bn_ema.contiguous().view(-1, feats_bn_ema.size(-1))

                feats_af = feats_af.view(device_num, -1, feats_af.size(-1))
                src_feat_af, tgt_feat_af = feats_af.split(feats_af.size(1) // 2, dim=1)
                src_feat_af, tgt_feat_af = src_feat_af.contiguous().view(-1, feats_af.size(-1)), tgt_feat_af.contiguous().view(-1, feats_af.size(-1))

                feats_af_ema = feats_af_ema.view(device_num, -1, feats_af_ema.size(-1))
                src_feat_af_ema, tgt_feat_af_ema = feats_af_ema.split(feats_af_ema.size(1) // 2, dim=1)
                src_feat_af_ema, tgt_feat_af_ema = src_feat_af_ema.contiguous().view(-1, feats_af_ema.size(-1)), tgt_feat_af_ema.contiguous().view(-1, feats_af_ema.size(-1))

                # src loss
                loss_src_ce = self.criterion_ce(src_prec[:, :self.src_classes], src_labels, weight_src)
                loss_ce_src_meter.update(loss_src_ce.item())
                loss_src_tri = self.hard_tri(src_feat_bn, src_labels,weight_src)
                loss_tri_src_meter.update(loss_src_tri.item())
                # tgt loss
                loss_tgt_ce = self.criterion_ce(tgt_prec[:, self.src_classes:self.src_classes + num_cluster], tgt_labels, weight_tgt)
                loss_ce_tgt_meter.update(loss_tgt_ce.item())
                loss_tgt_tri = self.hard_tri(tgt_feat_bn, tgt_labels,weight_tgt)
                loss_tri_tgt_meter.update(loss_tgt_tri.item())

                loss = loss_src_ce + loss_src_tri + loss_tgt_ce + loss_tgt_tri

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                self._update_ema_variables(self.model, self.model_ema, self.beta, epoch * len(data_loader_target) + i)
                del s_inputs, t_inputs, feats_af, inputs, pres, feats_bn, src_prec, tgt_prec, src_feat_bn, tgt_feat_bn,loss_src_ce,loss_src_tri,loss_tgt_ce,loss_tgt_tri,loss

            if (i + 1) % 100 == 0:
                print('Epoch: [{}][{}/{}]\t' 
                      'ce src {:.3f}\t'
                      'triplet src{:.3f}\t'
                      'ce target {:.3f} \t'
                      'triplet target {:.3f}\t'

                      .format(epoch, i + 1, len(data_loader_target),
                              loss_ce_src_meter.avg,
                              loss_tri_src_meter.avg,
                              loss_ce_tgt_meter.avg,
                              loss_tri_tgt_meter.avg,
                              )
                      )
        del src_inputs,target_inputs,src_imgs,tgt_imgs,src_labels,tgt_labels,weight_src,weight_tgt

    def _update_ema_variables(self, model, ema_model, beta, global_step):
        beta = min(1 - 1 / (global_step + 1), beta)
        model_state = model.state_dict()
        model_ema_state = ema_model.state_dict()
        new_dict = {}
        for key in model_state:
            new_dict[key] = beta * model_ema_state[key] + (1 - beta) * model_state[key]
        ema_model.load_state_dict(new_dict)

    def _parse_data(self, inputs):
        img, fname, pid, camid, index,weight = inputs
        img = img.cuda()
        pid = pid.cuda()
        weight = weight.cuda()
        return img, pid, weight,fname

