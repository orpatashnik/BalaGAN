import os
import random

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid

from data import EncoderDataSetWrapper
from networks.encoder import ModalitiesEncoder
from utils import write_loss, Timer, get_model_list


class ModalitiesEncoderTrainer:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.model = ModalitiesEncoder(config).to(config['device'])
        lr = config['modalities_encoder']['lr']
        encoder_params = list(self.model.parameters())
        self.opt = torch.optim.Adam(encoder_params, lr=lr, weight_decay=config['modalities_encoder']['weight_decay'])
        self.data_loader = EncoderDataSetWrapper(config).get_data_loader()
        self.T_max = len(self.data_loader.dataset)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.T_max, eta_min=0, last_epoch=-1)

    def train(self, first_epoch=0):
        global_it = first_epoch * len(self.data_loader)
        num_epochs = self.config['modalities_encoder']['epochs']
        for epoch_counter in range(first_epoch, num_epochs):
            with Timer("Elapsed time for epoch: %f"):
                for it, ((xis, xjs), _, _) in enumerate(self.data_loader):
                    self.opt.zero_grad()

                    xis = xis.to(self.config['device'])
                    xjs = xjs.to(self.config['device'])

                    self.loss_enc_contrastive = self.model.calc_nt_xent_loss(xis, xjs)
                    self.loss_enc_contrastive.backward()
                    self.opt.step()

                    print("Epoch: {curr_epoch}/{total_epochs} | Iteration: {curr_iter}/{total_iter} | Loss: {curr_loss}"
                          .format(curr_epoch=epoch_counter + 1, total_epochs=num_epochs,
                                  curr_iter=str(global_it + 1).zfill(8),
                                  total_iter=str(len(self.data_loader) * num_epochs).zfill(8),
                                  curr_loss=self.loss_enc_contrastive))

                    # Logging loss
                    if global_it % self.config['logger']['log_loss'] == 0:
                        write_loss(global_it, self, self.logger)

                    global_it += 1

            if epoch_counter % self.config['logger']['checkpoint_modalities_encoder_every'] == 0:
                self.save(self.config['logger']['checkpoint_dir'], epoch_counter)

            if epoch_counter >= 10:
                self.scheduler.step()

    def save(self, snapshot_dir, epoch):
        name = os.path.join(snapshot_dir, 'enc_{ep}.pt'.format(ep=str(epoch).zfill(5)))
        encoder = {
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict()
        }
        torch.save(encoder, name)

    def load(self, checkpoint_dir):
        last_model_name = get_model_list(checkpoint_dir, "enc")
        last_epoch = int(last_model_name.split('_')[1].split('.')[0])
        encoder = torch.load(last_model_name)
        self.model.load_state_dict(encoder["model"])
        self.opt.load_state_dict(encoder["opt"])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.T_max, eta_min=0,
                                                                    last_epoch=last_epoch)
        print(f"Resume encoder from epoch {last_epoch + 1}")
        return last_epoch + 1



class ModalitiesExtractor:

    def __init__(self, config):
        self.config = config

    def _extract_embeddings(self, encoder, data_loader):
        path_to_embedding = {}
        encoder.eval()
        for inputs, _, paths in data_loader:
            with torch.no_grad():
                output, _ = encoder(inputs.to(self.config['device']))
            for idx, path in enumerate(paths):
                path_to_embedding[path] = output.data[idx].detach().cpu()
        encoder.train()
        return path_to_embedding


    def _cluster_embeddings(self, path_to_embedding, k, first_cluster_index=0):
        import faiss
        if len(path_to_embedding) == 0:
            return {}
        path_to_cluster = {}
        paths = path_to_embedding.keys()
        embeddings = np.array([path_to_embedding[p].numpy() for p in paths])
        kmeans = faiss.Kmeans(d=embeddings[0].shape[0], k=k, spherical=True)
        kmeans.seed = 1234
        kmeans.train(embeddings)
        labels = kmeans.assign(embeddings)[1]
        for i, path in enumerate(paths):
            path_to_cluster[path] = first_cluster_index + labels[i]
        return path_to_cluster

    def get_modalities(self, encoder, data_loader):
        source_dir_path = f"{self.config['data']['train_root']}/A"
        target_dir_path = f"{self.config['data']['train_root']}/B"
        source_domain_paths = [os.path.join(source_dir_path, p) for p in os.listdir(source_dir_path)]
        target_domain_paths = [os.path.join(target_dir_path, p) for p in os.listdir(target_dir_path)]
        path_to_embedding = self._extract_embeddings(encoder, data_loader)
        source_domain_embeddings = dict((p, path_to_embedding[p]) for p in source_domain_paths)
        target_domain_embeddings = dict((p, path_to_embedding[p]) for p in target_domain_paths)
        source_modalities = self._cluster_embeddings(source_domain_embeddings, self.config['modalities']['k_source'])
        target_modalities = self._cluster_embeddings(target_domain_embeddings, self.config['modalities']['k_target'],
                                                     first_cluster_index=self.config['modalities']['k_source'])
        modalities = {**source_modalities, **target_modalities}
        return modalities

    def get_modalities_grid_image(self, modalities, images_per_modality=10):
        num_modalities = len(set(modalities.values()))
        grid = torch.zeros([num_modalities * images_per_modality, 3,
                            self.config['data']['crop_image_height'], self.config['data']['crop_image_width']])
        transform = transforms.Compose([transforms.Resize(self.config['data']['new_size']),
                                        transforms.RandomCrop((self.config['data']['crop_image_height'],
                                                               self.config['data']['crop_image_width']),
                                                               pad_if_needed=True),
                                        transforms.ToTensor()])
        for m_index, modality in enumerate(set(modalities.values())):
            images_paths = [im_path for im_path in modalities if modalities[im_path] == modality]
            sampled_images_paths = random.sample(images_paths, images_per_modality)
            for i_index, im_path in enumerate(sampled_images_paths):
                im = Image.open(im_path)
                im = transform(im)
                grid[m_index + num_modalities * i_index] = im
        return make_grid(grid, nrow=num_modalities)




