import copy

import torch
import torch.nn as nn

from networks.GAN import Generator, Discriminator


def recon_criterion(predict, target):
    return torch.mean(torch.abs(predict - target))


class BalaGAN(nn.Module):
    def __init__(self, config):
        super(BalaGAN, self).__init__()
        self.config = config
        self.gen = Generator(config)
        self.dis = Discriminator(config)
        self.gen_test = copy.deepcopy(self.gen)

    def forward(self, source_data, target_data, hp, mode):
        xa = source_data[0].to(self.config['device'])
        la = source_data[1].to(self.config['device'])
        xb = target_data[0].to(self.config['device'])
        lb = target_data[1].to(self.config['device'])

        if mode == 'gen_update':
            c_xa = self.gen.enc_content(xa)
            s_xa = self.gen.enc_class_model(xa)
            s_xb = self.gen.enc_class_model(xb)
            xt = self.gen.decode(c_xa, s_xb)  # translation
            xr = self.gen.decode(c_xa, s_xa)  # reconstruction
            l_adv_t, gacc_t, xt_gan_feat = self.dis.calc_gen_loss(xt, lb)
            l_adv_r, gacc_r, xr_gan_feat = self.dis.calc_gen_loss(xr, la)
            _, xb_gan_feat, dis_label_b, _, _ = self.dis(xb, lb)
            _, xa_gan_feat, dis_label_a, _, _ = self.dis(xa, la)
            l_c_rec = recon_criterion(xr_gan_feat.mean(3).mean(2),
                                      xa_gan_feat.mean(3).mean(2))
            l_m_rec = recon_criterion(xt_gan_feat.mean(3).mean(2),
                                      xb_gan_feat.mean(3).mean(2))
            l_x_rec = recon_criterion(xr, xa)
            l_adv = 0.5 * (l_adv_t + l_adv_r)
            acc = 0.5 * (gacc_t + gacc_r)
            l_total = (hp['gan']['gan_w'] * l_adv + hp['gan']['r_w'] * l_x_rec + hp['gan'][
                'fm_w'] * (l_c_rec + l_m_rec))
            l_total.backward()
            return l_total, l_adv, l_x_rec, l_c_rec, l_m_rec, acc

        elif mode == 'dis_update':
            xb.requires_grad_()
            l_real_pre, acc_r, resp_r = self.dis.calc_dis_real_loss(xb, lb)
            l_cls_pre, _ = self.dis.calc_dis_cls_loss(xb, lb)
            l_real = hp['gan']['gan_w'] * l_real_pre
            l_cls = hp['gan']['gan_w'] * l_cls_pre
            l_real.backward(retain_graph=True)
            l_cls.backward(retain_graph=True)
            l_reg_pre = self.dis.calc_grad2(resp_r, xb)
            l_reg = 10 * l_reg_pre
            l_reg.backward()
            with torch.no_grad():
                c_xa = self.gen.enc_content(xa)
                s_xb = self.gen.enc_class_model(xb)
                xt = self.gen.decode(c_xa, s_xb)
            l_fake_p, acc_f, resp_f = self.dis.calc_dis_fake_loss(xt.detach(),
                                                                  lb)
            l_fake = hp['gan']['gan_w'] * l_fake_p
            l_fake.backward()
            l_total = l_fake + l_real + l_reg
            acc = 0.5 * (acc_f + acc_r)
            return l_total, l_fake_p, l_real_pre, l_reg_pre, acc, l_cls
        else:
            assert 0, 'Not support operation'

    def test(self, source_data, target_data):
        self.eval()
        self.gen.eval()
        self.gen_test.eval()
        xa = source_data[0].to(self.config['device'])
        xb = target_data[0].to(self.config['device'])
        c_xa_current = self.gen.enc_content(xa)
        s_xa_current = self.gen.enc_class_model(xa)
        s_xb_current = self.gen.enc_class_model(xb)
        xt_current = self.gen.decode(c_xa_current, s_xb_current)
        xr_current = self.gen.decode(c_xa_current, s_xa_current)
        c_xa = self.gen_test.enc_content(xa)
        s_xa = self.gen_test.enc_class_model(xa)
        s_xb = self.gen_test.enc_class_model(xb)
        xt = self.gen_test.decode(c_xa, s_xb)
        xr = self.gen_test.decode(c_xa, s_xa)
        self.train()
        return xa, xr_current, xt_current, xb, xr, xt


    def translate_k_shot(self, source_data, target_data, k):
        self.eval()
        xa = source_data[0].to(self.config['device'])
        xb = target_data[0].to(self.config['device'])
        c_xa_current = self.gen_test.enc_content(xa)
        if k == 1:
            c_xa_current = self.gen_test.enc_content(xa)
            s_xb_current = self.gen_test.enc_class_model(xb)
            xt_current = self.gen_test.decode(c_xa_current, s_xb_current)
        else:
            s_xb_current_before = self.gen_test.enc_class_model(xb)
            s_xb_current_after = s_xb_current_before.squeeze(-1).permute(1,
                                                                         2,
                                                                         0)
            s_xb_current_pool = torch.nn.functional.avg_pool1d(
                s_xb_current_after, k)
            s_xb_current = s_xb_current_pool.permute(2, 0, 1).unsqueeze(-1)
            xt_current = self.gen_test.decode(c_xa_current, s_xb_current)
        return xt_current

    def compute_style(self, style_batch, k=1):
        self.eval()
        style_batch = style_batch.to(self.config['device'])
        s_xb_before = self.gen_test.enc_class_model(style_batch)
        s_xb_after = s_xb_before.squeeze(-1).permute(1, 2, 0)
        s_xb_pool = torch.nn.functional.avg_pool1d(s_xb_after, k)
        s_xb = s_xb_pool.permute(2, 0, 1).unsqueeze(-1)
        return s_xb

    def translate_simple(self, content_image, class_code):
        self.eval()
        xa = content_image.to(self.config['device'])
        s_xb_current = class_code.to(self.config['device'])
        c_xa_current = self.gen_test.enc_content(xa)
        xt_current = self.gen_test.decode(c_xa_current, s_xb_current)
        return xt_current
