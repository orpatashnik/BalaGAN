import argparse
import os

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from gan_trainer import GANTrainer
from utils import get_config, get_test_gan_loaders


def run_inference(config):
    source_loader, target_loader = get_test_gan_loaders(config, 10)
    gan_trainer = GANTrainer(config)
    gan_trainer.to(config['device'])
    gan_trainer.load_ckpt(config['ckpt'])
    gan_trainer.eval()

    for (source, target) in zip(source_loader, target_loader):
        source_im = source[0]
        source_path = source[1]
        target_im = target[0]
        target_path = target[1]
        with torch.no_grad():
            target_code = gan_trainer.model.compute_style(target_im)
            output_image = gan_trainer.model.translate_simple(source_im, target_code)
        for i in range(output_image.shape[0]):
            output_path = os.path.join(config['results'], f"{source_path[i].split('/')[-1]}_{target_path[i].split('/')[-1]}.png")
            save_image(output_image[i], output_path, normalize=True)



def create_grid(config):
    gan_trainer = GANTrainer(config)
    gan_trainer.to(config['device'])
    gan_trainer.load_ckpt(config['ckpt'])
    gan_trainer.eval()

    transform_list = [transforms.Resize((config['data']['new_size'])),
                      transforms.CenterCrop((config['data']['crop_image_height'], config['data']['crop_image_width'])),
                      transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transform = transforms.Compose(transform_list)

    source_images = os.listdir(os.path.join(opts.grid_dir, "A"))
    target_images = os.listdir(os.path.join(opts.grid_dir, "B"))

    grid = torch.zeros(1, 3, config['data']['crop_image_height'], config['data']['crop_image_width'])
    for j, path in enumerate(source_images):
        im_s_full_path = os.path.join(opts.grid_dir, "A", path)
        im_s = Image.open(im_s_full_path).convert('RGB')
        source_im_tensor = transform(im_s).unsqueeze(0)
        grid = torch.cat([grid, source_im_tensor], dim=0)

    for i, path in enumerate(target_images):
        im_t_full_path = os.path.join(opts.grid_dir, "B", path)
        im_t = Image.open(im_t_full_path).convert('RGB')
        target_im_tensor = transform(im_t).unsqueeze(0).cuda()
        grid = torch.cat([grid.cuda(), target_im_tensor], dim=0)
        with torch.no_grad():
            target_code = gan_trainer.model.compute_style(target_im_tensor)

        for j, path in enumerate(source_images):
            im_s_full_path = os.path.join(opts.grid_dir, "A", path)
            im_s = Image.open(im_s_full_path).convert('RGB')
            source_im_tensor = transform(im_s).unsqueeze(0)

            with torch.no_grad():
                output_image = gan_trainer.model.translate_simple(source_im_tensor, target_code)
                grid = torch.cat([grid, output_image], dim=0)

    save_image(grid.cpu(), os.path.join(config['results'], "grid.png"), nrow=len(source_images) + 1, normalize=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/dog2wolf.yaml', help='configuration file for training and testing')
    parser.add_argument('--use_cpu', default=False, action='store_true', help='using CPU instead of GPU')
    parser.add_argument('--results_path', type=str, default='results', help='Outputs path root')
    parser.add_argument('--gen_checkpoint', type=str)
    parser.add_argument('--create_grid', action='store_true')
    parser.add_argument('--grid_dir', type=str)
    parser.add_argument('--gpu_index', type=int, default=0, help='GPU index that will be used in the run')

    opts = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(opts.gpu_index)
    os.makedirs(opts.results_path, exist_ok=True)
    config = get_config(opts.config)
    config['device'] = torch.device("cpu") if opts.use_cpu else torch.device("cuda")
    config['results'] = opts.results_path
    config['ckpt'] = opts.gen_checkpoint

    run_inference(config)

    if opts.create_grid:
        create_grid(config)

