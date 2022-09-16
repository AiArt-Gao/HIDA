import torch
from os import path as osp, makedirs
from numpy import random
from PIL import Image
import numpy as np


def soft2num(label):
    label = label.permute(0, 2, 3, 1)
    label = torch.max(label, 3)[1]
    return label.unsqueeze(1).float()

def tensor_to_3(par):
    print("par.size:"+str(par.size()))
    b, c, h, w = par.size()

    img = torch.zeros(b, 3, h, w)
    img[:,0:2,:,:] = par
    img[:,2,:,:] = par[:,1,:,:]
    return img
def par_tensor2pix(label, par_dim, one_hot=True, norm=True):
    '''
        Label Content:
        0:         1:face       2:left eyebrow  3:right eyebrow 4:           5:
        6: eye     7:left ear   8: right ear    9:              10:noses     11:
        12:up lip  13:down lip  14:neak         15:             16:clothes   17:hair
        18: 19: 20: 21: 22: 23: 24:
        0[255,255,255]  1[255, 85, 0] 2[255, 170, 0]  3[255, 0, 85]  4[255, 0, 170]  5[0, 255, 0]
        6  7  8 10 12 13 14 16 17
        '''
    label = label[:, :par_dim]
    if one_hot:
        label = soft2num(label).squeeze(1)
    else:
        label = label.permute([0, 2, 3, 1])
        label = label.argmax(dim=-1)
        if len(label.size()) == 2:
            label = label.unsqueeze(0)
    # rgb_list = torch.FloatTensor(
    #     [[169, 209, 142], [181, 215, 243], [128, 64, 128], [128, 64, 128], [0, 0, 0], [0, 0, 0], [153, 153, 153],
    #      [0, 24, 179], [255, 128, 255], [0, 24, 179], [76, 110, 155]
    #         , [140, 181, 241], [172, 58, 43], [42, 49, 32], [162, 0, 163], [228, 165, 0], [66, 214, 109],
    #      [148, 195, 252], [151, 34, 176]]).to(label.device)
    rgb_list = torch.FloatTensor([[255, 0, 0], [255, 85, 0], [255, 170, 0],
                                  [255, 0, 85], [255, 0, 170],
                                  [0, 255, 0], [85, 255, 0], [170, 255, 0],
                                  [0, 255, 85], [0, 255, 170],
                                  [0, 0, 255], [85, 0, 255], [170, 0, 255],
                                  [0, 85, 255], [0, 170, 255],
                                  [255, 255, 0], [255, 255, 85], [255, 255, 170],
                                  [255, 0, 255], [255, 85, 255], [255, 170, 255],
                                  [0, 255, 255], [85, 255, 255], [170, 255, 255]]).to(label.device)

    b, h, w = label.size()

    img = torch.zeros(b, h, w, 3, device=label.device)
    img[label == 0] = rgb_list[0]
    img[label == 1] = rgb_list[1]  # 面部
    img[label == 2] = rgb_list[2]
    img[label == 3] = rgb_list[3]
    img[label == 4] = rgb_list[4]  # 左眼
    img[label == 5] = rgb_list[5]  # 右眼
    img[label == 6] = rgb_list[6]  # 眼镜维度
    img[label == 7] = rgb_list[7]
    img[label == 8] = rgb_list[8]
    img[label == 9] = rgb_list[9]
    img[label == 10] = rgb_list[10]
    img[label == 11] = rgb_list[11]
    img[label == 12] = rgb_list[12]

    img[label == 13] = rgb_list[13]
    img[label == 14] = rgb_list[14]
    img[label == 15] = rgb_list[15]
    img[label == 16] = rgb_list[16]
    img[label == 17] = rgb_list[17]
    img[label == 18] = rgb_list[18]

    if norm:
        img = img / 255
    img = img.permute([0, 3, 1, 2])
    return img


def save_imgarrs_to_disk(np_arrs, root, id_list=None):
    '''
        进入一个带bsize的nparr。根据idlist中的文件名来在固定位置保存图片。id list为空时候，就按1开始编号
    :param np_arrs: bsize的nparr
    :param root: 图片存储的目标路径
    :param id_list: 文件名list
    :return:
    '''
    if not osp.exists(root):
        makedirs(root)
    img_size = np_arrs.shape[0]
    if id_list is not None:
        assert img_size == len(id_list)
    else:
        id_list = [str(i + 1) for i in range(img_size)]
    for i in range(img_size):
        name = id_list[i]
        save_path = osp.join(root, name + '.jpg')
        pi_img = Image.fromarray(np.uint8(np_arrs[i] * 255), mode='RGB')
        pi_img.save(save_path)


def transfer_onechan_grey_to_three(onec_np_arr):
    '''
    一通道灰度转三通道灰度
    :param onec_np_arr:
    :return:
    '''
    if isinstance(onec_np_arr, torch.Tensor):
        onec_np_arr = onec_np_arr.detach().cpu().numpy()
    return np.concatenate([onec_np_arr, onec_np_arr, onec_np_arr], axis=1)  # channel维度


def select_number_of_images(arr, num_of_img):
    assert len(arr.shape) is 4
    arr_size = arr.shape[0]
    idx = random.randint(0, arr_size, size=num_of_img)
    return arr[idx], idx
