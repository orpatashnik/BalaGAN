import torch
import os
import argparse
import shutil

from tensorboardX import SummaryWriter

from extract_modalities import ModalitiesExtractor, ModalitiesEncoderTrainer
from utils import get_config, make_result_folders, save_image_tb, get_gan_loaders, get_exp_dir_name
from utils import write_loss, write_1images, Timer, get_modalities_extraction_loader
from gan_trainer import GANTrainer
import torch.backends.cudnn as cudnn
# Enable auto-tuner to find the best algorithm to use for your hardware.
cudnn.benchmark = True


def main(config, logger):

    print("Start extracting modalities...\n")

    modalities_encoder_trainer = ModalitiesEncoderTrainer(config, logger)
    encoder_first_epoch = modalities_encoder_trainer.load(config['logger']['checkpoint_dir']) if config['resume'] else 0
    modalities_encoder_trainer.train(encoder_first_epoch)

    modalities_extraction_loader = get_modalities_extraction_loader(config)
    modalities_extractor = ModalitiesExtractor(config)
    modalities = modalities_extractor.get_modalities(modalities_encoder_trainer.model, modalities_extraction_loader)
    modalities_grid = modalities_extractor.get_modalities_grid_image(modalities)
    logger.add_image("modality_per_col", modalities_grid, 0)

    del modalities_encoder_trainer
    del modalities_extractor
    torch.cuda.empty_cache()

    print("Finished extracting modalities, begin training the translation network...\n")

    train_source_loader, train_ref_loader, test_source_loader, test_ref_loader = get_gan_loaders(config, modalities)
    gan_trainer = GANTrainer(config)
    gan_trainer.to(config['device'])

    global_it = gan_trainer.resume(config['logger']['checkpoint_dir'], config) if config['resume'] else 0
    while global_it < config['gan']["max_iter"]:
        for it, (source_data, ref_data) in enumerate(zip(train_source_loader, train_ref_loader)):
            with Timer("Elapsed time in update: %f"):
                d_acc = gan_trainer.dis_update(source_data, ref_data, config)
                g_acc = gan_trainer.gen_update(source_data, ref_data, config)

                torch.cuda.synchronize(config['device'])

                print('D acc: %.4f\t G acc: %.4f' % (d_acc, g_acc))
                print("Iteration: {curr_iter}/{total_iter}"
                      .format(curr_iter=str(global_it + 1).zfill(8), total_iter=str(config['gan']['max_iter']).zfill(8)))

            # Save images for evaluation
            if global_it % config['logger']['eval_every'] == 0:
                with torch.no_grad():
                    for (val_source_data, val_ref_data) in zip(train_source_loader, train_ref_loader):
                        val_image_outputs = gan_trainer.test(val_source_data, val_ref_data)
                        write_1images(val_image_outputs, config['logger']['image_dir'], 'train_{iter}'
                                      .format(iter=global_it))
                        save_image_tb(val_image_outputs, "train", global_it, logger)
                        break
                    for (test_source_data, test_ref_data) in zip(test_source_loader, test_ref_loader):
                        test_image_outputs = gan_trainer.test(test_source_data, test_ref_data)
                        write_1images(test_image_outputs, config['logger']['image_dir'], 'test_{iter}'
                                      .format(iter=global_it))
                        save_image_tb(test_image_outputs, "test", global_it, logger)
                        break

            # Log losses
            if global_it % config['logger']['log_loss'] == 0:
                write_loss(global_it, gan_trainer, logger)

            # Save checkpoint
            if global_it % config['logger']['checkpoint_gan_every'] == 0:
                gan_trainer.save(config['logger']['checkpoint_dir'], global_it)

            global_it += 1

    print("Finished training!")


def setup(opts):

    # configure GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opts.gpu_index)

    # load experiment setting
    config = get_config(opts.config)

    # setup logger and output folders
    output_directory = get_exp_dir_name(opts.output_path, opts.exp_name, opts.resume)
    checkpoint_directory, image_directory, logs_directory = make_result_folders(output_directory)
    writer = SummaryWriter(logs_directory)
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))

    # setup config
    config['resume'] = opts.resume
    config['device'] = torch.device("cpu") if opts.use_cpu else torch.device("cuda")
    config['logger']['logs_dir'] = logs_directory
    config['logger']['checkpoint_dir'] = checkpoint_directory
    config['logger']['image_dir'] = image_directory

    return config, writer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/dog2wolf.yaml', help='configuration file for training and testing')
    parser.add_argument('--use_cpu', default=False, action='store_true', help='using CPU instead of GPU')
    parser.add_argument('--output_path', type=str, default='.', help='Outputs path root')
    parser.add_argument('--resume', action='store_true', help='Use this flag to resume an existing experiment')
    parser.add_argument('--exp_name', type=str, help='The name of the directory that will be used for this run. '
                                                     'If resuming an existing experiment, the latest experiment '
                                                     'with this name will be loaded.')
    parser.add_argument('--gpu_index', type=int, default=0, help='GPU index that will be used in the run')

    opts = parser.parse_args()

    config, logger = setup(opts)
    main(config, logger)
