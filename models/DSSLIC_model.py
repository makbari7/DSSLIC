### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from PIL import Image
import pytorch_msssim
import util.util as util
import os
import time
import torchvision.transforms as transforms

from data.base_dataset import get_transform

class DSSLICModel(BaseModel):
    def name(self):
        return 'DSSLICModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none': # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        input_nc = opt.label_nc if opt.label_nc != 0 else 3

        ##### define networks        
        # Generator network
        if self.opt.no_seg:
            input_nc = 0        
        netG_input_nc = input_nc 
        if self.opt.comp_type != 'none':
            netG_input_nc += 3 # for downsampling | compG
        if not opt.no_instance:
            netG_input_nc += 1
        
        self.compG = networks.define_compG(netG_input_nc, opt.output_nc, opt.ncf, opt.n_downsample_comp, norm=opt.norm, gpu_ids=self.gpu_ids)  
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, 
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)        

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.output_nc
            if not opt.no_instance:
                netD_input_nc += 1
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
            
        print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            self.load_network(self.compG, 'C', opt.which_epoch, pretrained_path)                        
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)  

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)   
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:             
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)
        
            # Names so we can breakout loss
            self.loss_names = ['G_GAN', 'G_GAN_Feat', 'G_VGG' , 'G_DIS', 'G_SSIM' ,'D_real', 'D_fake']            

            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():       
                    if key.startswith('model' + str(opt.n_local_enhancers)):
                        params += [{'params':[value],'lr':opt.lr}]
                    else:
                        params += [{'params':[value],'lr':0.0}]                            
            else:
                params = list(self.netG.parameters())
                params += list(self.compG.parameters())                

            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))                            

            # optimizer D                        
            params = list(self.netD.parameters())    
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))


    def encode_input(self, flabel_map=None, real_image=None, ds_image=None, infer=False):
        input_flabel = None        
        if not self.opt.no_seg: # seg             
          if self.opt.label_nc == 0:              
              if flabel_map is not None:
                input_flabel = flabel_map.data.cuda()
          else:
              # create one-hot vector for label map 
              size = flabel_map.size()
              oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
              if flabel_map is not None:
                input_flabel = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
                input_flabel = input_flabel.scatter_(1, flabel_map.data.long().cuda(), 1.0)            
            
        if flabel_map is not None:
          input_flabel = Variable(input_flabel, volatile=infer)

        # real images for training
        if real_image is not None:
            real_image = Variable(real_image.data.cuda())
        if ds_image is not None:
            ds_image = Variable(ds_image.data.cuda())

        return input_flabel, real_image, ds_image

    def discriminate(self, input_label, test_image, use_pool=False):
        if input_label is not None:
          input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        else:
          input_concat = test_image.detach()
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    # flabel: Fake Label (psp output)
    def forward(self, flabel, image, ds_image, infer=False):
        # Encode Inputs
        input_flabel, real_image, ds_image = self.encode_input(flabel, image, ds_image)  

        # input to G: downsampled | compact image
        if self.opt.comp_type=='compG': # use compG
          if not self.opt.no_seg: # with seg
            compG_input = torch.cat((input_flabel, real_image), dim=1)                                                                                                 
          else: # no seg
            compG_input = real_image;
          comp_image = self.compG.forward(compG_input)

          ### tensor-level bilinear
          upsample = torch.nn.Upsample(scale_factor=self.opt.alpha, mode='bilinear')      
          up_image = upsample(comp_image)      

        else: # use bicubic downsampling (ds)
          up_image = ds_image
          
        if not self.opt.no_seg: # seg
          if self.opt.comp_type!='none': # seg, ds | comp_image
            input_fconcat = torch.cat((input_flabel, up_image), dim=1)
          else: # no ds, but seg
            input_fconcat = input_flabel
        else: # no seg
          input_flabel = None
          if self.opt.comp_type != 'none': # ds (ds | comp_image)
            input_fconcat = up_image                        

        # add compact image, so that G tries to find the best residual
        res = self.netG.forward(input_fconcat)
        fake_image_f = res + up_image

        # Fake Detection and Loss        
        pred_fake_pool_f = self.discriminate(input_flabel, fake_image_f, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool_f, False)

        # Real Detection and Loss                
        pred_real_f = self.discriminate(input_flabel, real_image)
        loss_D_real = self.criterionGAN(pred_real_f, True)

        # GAN loss (Fake Passability Loss)
        if input_flabel is not None:
          inputD_concat = torch.cat((input_flabel, fake_image_f), dim=1)
        else:
          inputD_concat = fake_image_f        
        pred_fake_f = self.netD.forward(inputD_concat)
        loss_G_GAN = self.criterionGAN(pred_fake_f, True)
        
        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake_f[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake_f[i][j], pred_real_f[i][j].detach()) * self.opt.lambda_feat
                   
        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = (self.criterionVGG(fake_image_f, real_image)) * self.opt.lambda_feat            
            
        # l1 loss between x and x'
        loss_G_DIS = 0
        criterionDIS = torch.nn.L1Loss()
        loss_G_DIS = criterionDIS(fake_image_f, real_image) * self.opt.lambda_feat * 2
                
        # SSIM Loss
        loss_G_SSIM=0
        ssim_loss = pytorch_msssim.SSIM()
        loss_G_SSIM = -ssim_loss(real_image, fake_image_f)

        # Only return the fake_B image if necessary to save BW
        return [ [ loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_G_DIS, loss_G_SSIM, loss_D_real, loss_D_fake ], None if not infer else real_image, None if not infer else input_flabel, None if not infer else fake_image_f, None if not infer else res, None if not infer else comp_image, None if not infer else up_image ]        
    
    def inference(self, real, flabel, ds):
        # Encode Inputs                
        input_flabel, real_image, ds = self.encode_input(flabel_map=Variable(flabel), real_image=Variable(real), ds_image=Variable(ds), infer=True)

        # input to G: DS or compG output
        if self.opt.comp_type=='compG': # use compG
          if not self.opt.no_seg: # seg
            compG_input = torch.cat((input_flabel, real_image), dim=1)
          else: # no seg
            compG_input = real_image;
          
          if len(self.opt.gpu_ids)==0: # cpu
            compG_input = compG_input.cpu()
            input_flabel = input_flabel.cpu()            
            real_image = real_image.cpu()

          start_time = time.time()
          comp_image = self.compG.forward(compG_input)
          print("--- CompNet: %s seconds ---" % (time.time() - start_time))          
                                
          upsample = torch.nn.Upsample(scale_factor=self.opt.alpha, mode='bilinear')
          up_image = upsample(comp_image)

          # resize the input_label to match the size of upscaled compImage
          hw = list(up_image.size())
          upsample = torch.nn.Upsample(size=(hw[2],hw[3]), mode='bilinear')          
          input_flabel = upsample(input_flabel)

        else: # use bicubic ds
          up_image = ds
          comp_image = ds
        
        ### Fake Generation
        input_fconcat = input_flabel
        
        # concat ds image
        if not self.opt.no_seg: # seg
          if self.opt.comp_type!='none': # seg, ds            
            input_fconcat = torch.cat((input_flabel, up_image), dim=1)
          else: # no ds, but seg            
            input_fconcat = input_flabel
        else: # no seg
          if self.opt.comp_type!='ds': # ds            
            input_fconcat = up_image
        
        res_image = self.netG.forward(input_fconcat)
        fake_image = res_image + up_image
        print("--- RecNet: %s seconds ---" % (time.time() - start_time))
        
        return fake_image, res_image, comp_image, up_image

    def save(self, which_epoch):
    self.save_network(self.compG, 'C', which_epoch, self.gpu_ids)
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)        
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())           
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999)) 
        print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr




