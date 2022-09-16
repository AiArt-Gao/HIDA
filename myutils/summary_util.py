import os
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from os.path import exists
from os import path as osp
import numpy as np
from tqdm import tqdm


def tensor_to_image(tensor, zero2one=True):
    """
        将-1~1的tensor转成0~1(给tf logger)或者 0~255(给输出)的图片
    :param tensor: 输入的-1~1的tensor
    :param zero2one: 如果为False 范围为0~255
    :return:
    """
    tensor = (tensor + 1) / 2
    if not zero2one:
        tensor = tensor * 255
    return tensor


def loggger_group_image(logger: SummaryWriter, step, fake_img, org_img, target_img, tag):
    fake_img = tensor_to_image(fake_img)
    org_img = tensor_to_image(org_img)
    target_img = tensor_to_image(target_img)
    logger.add_images(tag="{}/gen_img".format(tag), img_tensor=fake_img, global_step=step)
    logger.add_images(tag="{}/org_img".format(tag), img_tensor=org_img, global_step=step)
    logger.add_images(tag="{}/target_img".format(tag), img_tensor=target_img, global_step=step)


def loggger_list_image(logger: SummaryWriter, step, img_list, pre_tag, tag_list):
    assert len(img_list) == len(tag_list)
    for i in range(len(img_list)):
        t_img = tensor_to_image(img_list[i])
        t_tag = "{}/{}".format(pre_tag, tag_list[i])
        logger.add_images(tag=t_tag, img_tensor=t_img, global_step=step)


def save_tensor_to_disk(path, img_list, dim=3):
    if not exists(path):
        os.makedirs(path, mode=0o755, exist_ok=True)

    # catting
    total = torch.cat(img_list, dim=dim)
    # transpose
    total = torch.transpose(total, 1, 3)
    total = torch.transpose(total, 1, 2)
    # -1~1 to 255
    total = tensor_to_image(total, zero2one=False)
    # to np
    total = total.detach().cpu().numpy()
    # to save
    for i in tqdm(range(total.shape[0])):
        t_img = total[i]
        name = str(i + 1)
        save_path = osp.join(path, name + '.jpg')
        pi_img = Image.fromarray(np.uint8(t_img), mode='RGB')
        pi_img.save(save_path)
