import os
import yaml
import time

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.utils as vutils

from data import ImageFolderWithPaths, SingleFolderDataset
from torch.utils.data.sampler import SubsetRandomSampler


def update_average(model_tgt, model_src, beta=0.999):
    with torch.no_grad():
        param_dict_src = dict(model_src.named_parameters())
        for p_name, p_tgt in model_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert(p_src is not p_tgt)
            p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)

def get_modalities_extraction_loader(config):
    transform = transforms.Compose([transforms.Resize(config['data']['new_size']),
                                    transforms.CenterCrop((config['data']['crop_image_height'], config['data']['crop_image_width'])),
                                    transforms.ToTensor()])
    dataset = ImageFolderWithPaths(root=config['data']['train_root'], transform=transform)
    sampler = SubsetRandomSampler(range(len(dataset)))
    return DataLoader(dataset, batch_size=config['gan']['batch_size'], sampler=sampler, num_workers=config['data']['num_workers'],
                      drop_last=False, shuffle=False)

def get_gan_loaders(config, modalities):
    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.Resize(config['data']['new_size']),
                                    transforms.RandomCrop((config['data']['crop_image_height'], config['data']['crop_image_width']),
                                                          pad_if_needed=True),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = ImageFolderWithPaths(custom_labels=modalities, root=config['data']['train_root'], transform=transform)
    test_dataset = ImageFolderWithPaths(root=config['data']['test_root'], transform=transform)
    train_content_loader = DataLoader(train_dataset, batch_size=config['gan']['batch_size'], num_workers=config['data']['num_workers'],
                                      drop_last=True, shuffle=True)
    train_style_loader = DataLoader(train_dataset, batch_size=config['gan']['batch_size'], num_workers=config['data']['num_workers'],
                                      drop_last=True, shuffle=True)
    test_content_loader = DataLoader(test_dataset, batch_size=config['gan']['batch_size'], num_workers=config['data']['num_workers'],
                                      drop_last=True, shuffle=True)
    test_style_loader = DataLoader(test_dataset, batch_size=config['gan']['batch_size'], num_workers=config['data']['num_workers'],
                                      drop_last=True, shuffle=True)
    return train_content_loader, train_style_loader, test_content_loader, test_style_loader

def get_test_gan_loaders(config, target_images_repeat=1):
    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.Resize(config['data']['new_size']),
                                    transforms.CenterCrop((config['data']['crop_image_height'], config['data']['crop_image_width'])),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    source_dataset = SingleFolderDataset(root=os.path.join(config['data']['test_root'], "A"), transform=transform)
    target_dataset = SingleFolderDataset(root=os.path.join(config['data']['test_root'], "B"), transform=transform,
                                         num_repeats=target_images_repeat)
    source_loader = DataLoader(source_dataset, batch_size=config['gan']['batch_size'], num_workers=config['data']['num_workers'], drop_last=True)
    target_loader = DataLoader(target_dataset, batch_size=config['gan']['batch_size'], num_workers=config['data']['num_workers'], drop_last=True)
    return source_loader, target_loader

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

def make_result_folders(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    logs_directory = os.path.join(output_directory, "logs")
    if not os.path.exists(logs_directory):
        print("Creating directory: {}".format(logs_directory))
        os.makedirs(logs_directory)
    return checkpoint_directory, image_directory, logs_directory

def __write_images(im_outs, dis_img_n, file_name):
    im_outs = [images.expand(-1, 3, -1, -1) for images in im_outs]
    image_tensor = torch.cat([images[:dis_img_n] for images in im_outs], 0)
    image_grid = vutils.make_grid(image_tensor.data,
                                  nrow=dis_img_n, padding=0, normalize=True)
    vutils.save_image(image_grid, file_name, nrow=1)

def __image_to_tb(im_outs, dis_img_n, name, iteration, writer):
    im_outs = [images.expand(-1, 3, -1, -1) for images in im_outs]
    image_tensor = torch.cat([images[:dis_img_n] for images in im_outs], 0)
    image_grid = vutils.make_grid(image_tensor.data,
                                  nrow=dis_img_n, padding=0, normalize=True)
    writer.add_image(name, image_grid, iteration)

def write_1images(image_outputs, image_directory, postfix):
    display_image_num = image_outputs[0].size(0)
    __write_images(image_outputs, display_image_num,
                   '%s/gen_%s.jpg' % (image_directory, postfix))

def save_image_tb(image_outputs, name, iteration, writer):
    display_image_num = image_outputs[0].size(0)
    __image_to_tb(image_outputs, display_image_num, name, iteration, writer)

def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer)
               if ((not callable(getattr(trainer, attr))
                    and not attr.startswith("__"))
                   and ('loss' in attr
                        or 'grad' in attr
                        or 'nwd' in attr
                        or 'accuracy' in attr))]
    for m in members:
        tag = m.split('_')[1]
        train_writer.add_scalar(f"{tag}/{m}", getattr(trainer, m), iterations)

def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and
                  key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    if len(gen_models) > 0:
        last_model_name = gen_models[-1]
    else:
        last_model_name = None
    return last_model_name

def get_exp_dir_name(output_path, exp_name, to_resume):
    exp_dir_path = os.path.join(output_path, "outputs", exp_name)
    if os.path.exists(exp_dir_path):
        exp_versions = [v for v in os.listdir(exp_dir_path) if str.isnumeric(v)]
        latest_version = max(exp_versions)
        version = latest_version if to_resume else to_resume + 1
    elif not to_resume:
        version = 0
    else:
        print("Warning: there is not experiment to resume, start training from scratch.")
        version = 0
    return os.path.join(exp_dir_path, str(version))

class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))
