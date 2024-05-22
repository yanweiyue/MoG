import os
import torch
import shutil
from collections import OrderedDict
import logging
import numpy as np


def save_ckpt(model, optimizer, loss, epoch, save_path, name_pre, name_post='best'):
    model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    state = {
            'epoch': epoch,
            'model_state_dict': model_cpu,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print("Directory ", save_path, " is created.")

    filename = '{}/{}_{}.pth'.format(save_path, name_pre, name_post)
    torch.save(state, filename)
    print('model has been saved as {}'.format(filename))


def load_pretrained_models(model, pretrained_model, phase, ismax=True):  # ismax means max best
    if ismax:
        best_value = -np.inf
    else:
        best_value = np.inf
    epoch = -1

    if pretrained_model:
        if os.path.isfile(pretrained_model):
            logging.info("===> Loading checkpoint '{}'".format(pretrained_model))
            checkpoint = torch.load(pretrained_model)
            try:
                best_value = checkpoint['best_value']
                if best_value == -np.inf or best_value == np.inf:
                    show_best_value = False
                else:
                    show_best_value = True
            except:
                best_value = best_value
                show_best_value = False

            model_dict = model.state_dict()
            ckpt_model_state_dict = checkpoint['state_dict']

            # rename ckpt (avoid name is not same because of multi-gpus)
            is_model_multi_gpus = True if list(model_dict)[0][0][0] == 'm' else False
            is_ckpt_multi_gpus = True if list(ckpt_model_state_dict)[0][0] == 'm' else False

            if not (is_model_multi_gpus == is_ckpt_multi_gpus):
                temp_dict = OrderedDict()
                for k, v in ckpt_model_state_dict.items():
                    if is_ckpt_multi_gpus:
                        name = k[7:]  # remove 'module.'
                    else:
                        name = 'module.'+k  # add 'module'
                    temp_dict[name] = v
                # load params
                ckpt_model_state_dict = temp_dict

            model_dict.update(ckpt_model_state_dict)
            model.load_state_dict(ckpt_model_state_dict)

            if show_best_value:
                logging.info("The pretrained_model is at checkpoint {}. \t "
                             "Best value: {}".format(checkpoint['epoch'], best_value))
            else:
                logging.info("The pretrained_model is at checkpoint {}.".format(checkpoint['epoch']))

            if phase == 'train':
                epoch = checkpoint['epoch']
            else:
                epoch = -1
        else:
            raise ImportError("===> No checkpoint found at '{}'".format(pretrained_model))
    else:
        logging.info('===> No pre-trained model')
    return model, best_value, epoch


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def process_indexes(idx_list):
    idx_dict = {}
    for i, idx in enumerate(idx_list):
        idx_dict[idx] = i

    return [idx_dict[i] for i in sorted(idx_dict.keys())]


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)