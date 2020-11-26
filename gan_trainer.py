import copy
import os
import math

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.optim import lr_scheduler

from balagan import BalaGAN
from utils import get_model_list


def update_average(model_tgt, model_src, beta=0.999):
    with torch.no_grad():
        param_dict_src = dict(model_src.named_parameters())
        for p_name, p_tgt in model_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert(p_src is not p_tgt)
            p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)


class GANTrainer(nn.Module):
    def __init__(self, config):
        super(GANTrainer, self).__init__()
        self.config = config
        self.model = BalaGAN(config)
        lr_gen = config['gan']['lr_gen']
        lr_dis = config['gan']['lr_dis']
        dis_params = list(self.model.dis.parameters())
        gen_params = list(self.model.gen.parameters())
        self.dis_opt = torch.optim.RMSprop(
            [p for p in dis_params if p.requires_grad], lr=lr_gen, weight_decay=config['gan']['weight_decay'])
        self.gen_opt = torch.optim.RMSprop(
            [p for p in gen_params if p.requires_grad], lr=lr_dis, weight_decay=config['gan']['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, config)
        self.gen_scheduler = get_scheduler(self.gen_opt, config)
        self.apply(weights_init(config['gan']['init']))
        self.model.gen_test = copy.deepcopy(self.model.gen)


    def gen_update(self, source_data, target_data, hp):
        self.gen_opt.zero_grad()
        al, ad, xr, cr, sr, ac = self.model(source_data, target_data, hp, 'gen_update')
        self.loss_gen_total = torch.mean(al)
        self.loss_gen_recon_x = torch.mean(xr)
        self.loss_gen_recon_c = torch.mean(cr)
        self.loss_gen_recon_s = torch.mean(sr)
        self.loss_gen_adv = torch.mean(ad)
        self.accuracy_gen_adv = torch.mean(ac)
        self.gen_opt.step()
        update_average(self.model.gen_test, self.model.gen)
        return self.accuracy_gen_adv.item()


    def dis_update(self, source_data, target_data, hp):
        self.dis_opt.zero_grad()
        al, lfa, lre, reg, acc, l_cls = self.model(source_data, target_data, hp, 'dis_update')
        self.loss_dis_total = torch.mean(al)
        self.loss_dis_fake_adv = torch.mean(lfa)
        self.loss_dis_real_adv = torch.mean(lre)
        self.loss_dis_reg = torch.mean(reg)
        self.accuracy_dis_adv = torch.mean(acc)
        self.loss_dis_cls = torch.mean(l_cls)
        self.dis_opt.step()
        return self.accuracy_dis_adv.item()

    def test(self, source_data, target_data):
        return self.model.test(source_data, target_data)

    def resume(self, checkpoint_dir, hp):
        last_model_name = get_model_list(checkpoint_dir, "gen")
        if last_model_name:
            state_dict = torch.load(last_model_name)
            self.model.gen.load_state_dict(state_dict['gen'])
            self.model.gen_test.load_state_dict(state_dict['gen_test'])
            iterations = int(last_model_name[-11:-3])

            last_model_name = get_model_list(checkpoint_dir, "dis")
            state_dict = torch.load(last_model_name)
            self.model.dis.load_state_dict(state_dict['dis'])

            last_opt_name = get_model_list(checkpoint_dir, "optimizer")
            state_dict = torch.load(last_opt_name)
            self.dis_opt.load_state_dict(state_dict['dis'])
            self.gen_opt.load_state_dict(state_dict['gen'])

            self.dis_scheduler = get_scheduler(self.dis_opt, hp, iterations)
            self.gen_scheduler = get_scheduler(self.gen_opt, hp, iterations)
        else:
            iterations = 0
        print(f'Resume GAN from iteration {iterations}')
        return iterations


    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations))
        opt_name = os.path.join(snapshot_dir, 'optimizer_%08d.pt' % (iterations))
        torch.save({'gen': self.model.gen.state_dict(),
                    'gen_test': self.model.gen_test.state_dict()}, gen_name)
        torch.save({'dis': self.model.dis.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(),
                    'dis': self.dis_opt.state_dict()}, opt_name)


    def load_ckpt(self, ckpt_name):
        state_dict = torch.load(ckpt_name)
        self.model.gen.load_state_dict(state_dict['gen'])
        self.model.gen_test.load_state_dict(state_dict['gen_test'])


    def load_dis_pt(self, pt_name):
        state_dic = torch.load(pt_name)
        self.model.dis.load_state_dict(state_dic['dis'])


    def translate(self, source_data, target_data):
        return self.model.translate(source_data, target_data)


    def forward(self, *inputs):
        print('Forward function not implemented.')
        pass


def get_scheduler(optimizer, hp, it=-1):
    if 'lr_policy' not in hp or hp['lr_policy'] == 'constant':
        scheduler = None  # constant scheduler
    elif hp['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hp['step_size'],
                                        gamma=hp['gamma'], last_epoch=it)
    else:
        return NotImplementedError('%s not implemented', hp['lr_policy'])
    return scheduler


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    return init_fun
