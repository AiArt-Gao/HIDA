#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-12-11 下午3:40
# @Author  : Jehovah
# @File    : pro_result.py
# @Software: PyCharm

import os
import cv2
import os
import PIL.Image as Image
from PIL import ImageDraw
import numpy as np


def cropImage():
    path = '/home/jehovah/PycharmProject/Image_enhance/sys_multihead_multi_2'
    # path = '/home/jehovah/PycharmProject/wnn_data/base'
    # path = '/home/jehovah/PycharmProject/B2A/base'
    savepath = path + '_200'

    new_width = 200
    new_height = 250
    i = 1
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    for root, _, files in os.walk(path):
        for file in sorted(files):
            image_root = os.path.join(root, file)
            im = Image.open(image_root)
            width, height = im.size  # Get dimensions

            left = (width - new_width) / 2
            top = (height - new_height) / 2
            right = (width + new_width) / 2
            bottom = (height + new_height) / 2

            im = im.crop((left, top, right, bottom))
            # im.save(savepath + '/' + str(i)+'.jpg', quality=100)
            # im.save(savepath + '/' + str(int(file.split('_')[0]) + 1) + '.jpg', quality=100)
            im.save(savepath + '/' + file, quality=100)
            i = i + 1


def catImage():
    coofoc = [
        # '/home/jehovah/PycharmProject/Image_enhance/photo',
        # '/home/jehovah/PycharmProject/Image_enhance/ground',
        # '/home/jehovah/PycharmProject/Image_enhance/base_1ch_200',
        # '/home/jehovah/PycharmProject/Image_enhance/multi_200',
        # # '/home/jehovah/PycharmProject/Image_enhance/ga_200',
        # # '/home/jehovah/PycharmProject/Image_enhance/sys_sys2_200',
        # # '/home/jehovah/PycharmProject/Image_enhance/sys_ga1_200',
        '/home/jehovah/PycharmProject/tatal/compare/ground/cuhk_b',
        # '/home/jehovah/PycharmProject/tatal/compare/ground/street_b',
        '/home/jehovah/PycharmProject/tatal/compare/practical',
        '/home/jehovah/PycharmProject/tatal/compare/final_practical',

        # '/home/jehovah/PycharmProjects/pix2pix/pytorch-CycleGAN-and-pix2pix-master/datasets/facades/testB',
        # '/home/jehovah/PycharmProject/pix2pix/facades',
        # '/home/jehovah/PycharmProject/pix2pix/sys_facades',
        # '/home/jehovah/PycharmProject/pix2pix/sys_facades_B2A_padding/200',

        # coofoc6 = '/home/jeh                      ovah/PycharmProject/Image_enhance/de2_sys2_200'
    ]
    savepath = '/home/jehovah/PycharmProject/tatal/cat3/cuhk'
    import os
    DIR = coofoc[0]  # 要统计的文件夹
    len_dir = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    UNIT_SIZE = 250  # 250
    UNIT_WIGHT = 200  # 200
    m_top = 5
    margin = 10
    TARGET_WIDTH = (UNIT_WIGHT + margin) * len(coofoc) + margin
    for i in range(0, len_dir):
        # to_resol = 256
        print(i)
        pipbe = []
        pip11be = Image.open(coofoc[0] + '/' + str(i + 1) + '.jpg').convert('RGB')
        pip12be = Image.open(coofoc[1] + '/' + str(i + 1) + '.jpg').convert('RGB')
        pipbe.append(pip11be)
        pipbe.append(pip12be)
        for k in range(2, len(coofoc)):
            pip13be = Image.open(coofoc[k] + '/' + str(i + 1) + '.jpg').convert('RGB')
            pipbe.append(pip13be)
        # pip16be = Image.open(coofoc[6] + '/' + str(i) + '_interpolation.jpg').convert('RGB')
        target = Image.new('RGB', (TARGET_WIDTH, UNIT_SIZE + m_top * 2), color='white')
        for k in range(len(pipbe)):
            if k == 0:
                target.paste(pipbe[k], (
                    (UNIT_WIGHT + margin) * k + margin, 0 + m_top, (UNIT_WIGHT) * (k + 1) + margin, UNIT_SIZE + m_top))
            else:
                target.paste(pipbe[k], (
                    (UNIT_WIGHT + margin) * k + margin, 0 + m_top, (UNIT_WIGHT + margin) * k + UNIT_WIGHT + margin,
                    UNIT_SIZE + m_top))

        target.save(savepath + '/' + str(i) + '.jpg', quality=100)


def catImage2():
    # coofoc = [
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE_Lsem/photo',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_final/eSPADE_Lsem_Lgeo_Dis/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_final/eSPADE_Lgeo_Lsem_Dis_combine_1/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_final/eSPADE_Lgeo_Lsem_Dis_combine_2/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE_Lsem/sketch',
    # ]
    coofoc = [
        '/home/lixiang/lx/SAND-pytorvh-master/output_apd/cyclegan/photo',
        '/home/lixiang/lx/SAND-pytorvh-master/output_apd/cyclegan/sketch',
        '/home/lixiang/lx/SAND-pytorvh-master/output_apd/eSPADE_Lgeo_Lsem_Lidn_0.5/x',
        '/home/lixiang/lx/SAND-pytorvh-master/output_apd/eSPADE_Lgeo_Lsem_Lidn_1/x',
        '/home/lixiang/lx/SAND-pytorvh-master/output_apd/eSPADE_Lgeo_Lsem_Lidn_2/x',
        '/home/lixiang/lx/SAND-pytorvh-master/output_apd/eSPADE_Lgeo_Lsem_Lidn_2.5/x',
        '/home/lixiang/lx/SAND-pytorvh-master/output_apd/eSPADE_Lgeo_Lsem_Lidn_3/x',
    ]
    DIR = coofoc[0]
    savepath = '/home/lixiang/lx/SAND-pytorvh-master/output_apd/cat_Lidn'
    # len_dir = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    UNIT_SIZE = 512  # 250
    UNIT_WIGHT = 512  # 200
    m_top = 5
    margin = 10
    # TARGET_WIDTH = (UNIT_WIGHT + margin) * (len(coofoc)/2) + margin
    TARGET_WIDTH = (UNIT_WIGHT + margin) * (len(coofoc)) + margin
    list = os.listdir(DIR)
    for name in list:
        # to_resol = 256
        print(name)
        pipbe = []
        pip11be = Image.open(coofoc[0] + '/' + name).convert('RGB')
        pip12be = Image.open(coofoc[1] + '/' + name).convert('RGB')
        pip13be = Image.open(coofoc[2] + '/' + name).convert('L')
        pip14be = Image.open(coofoc[3] + '/' + name).convert('L')
        pip15be = Image.open(coofoc[4] + '/' + name).convert('L')
        pip16be = Image.open(coofoc[5] + '/' + name).convert('L')
        pip17be = Image.open(coofoc[6] + '/' + name).convert('L')
        # pip18be = Image.open(coofoc[7] + '/' + name).convert('L')
        # pip19be = Image.open(coofoc[8] + '/' + name).convert('L')
        # pip110be = Image.open(coofoc[9] + '/' + name).convert('L')
        # pip111be = Image.open(coofoc[10] + '/' + name).convert('L')
        # pip112be = Image.open(coofoc[11] + '/' + name).convert('L')
        # pip113be = Image.open(coofoc[12] + '/' + name).convert('L')
        pipbe.append(pip11be)
        pipbe.append(pip12be)
        pipbe.append(pip13be)
        pipbe.append(pip14be)
        pipbe.append(pip15be)
        pipbe.append(pip16be)
        pipbe.append(pip17be)
        # pipbe.append(pip18be)
        # pipbe.append(pip19be)
        # pipbe.append(pip110be)
        # pipbe.append(pip111be)
        # pipbe.append(pip112be)
        # pipbe.append(pip113be)
        # for k in range(2, len(coofoc)):
        #     pip13be = Image.open(coofoc[k] + '/' + name).convert('L')
        #     pipbe.append(pip13be)
        # pip16be = Image.open(coofoc[6] + '/' + str(i) + '_interpolation.jpg').convert('RGB')
        target = Image.new('RGB', (TARGET_WIDTH, UNIT_SIZE + m_top * 2), color='white')
        for k in range(len(pipbe)):
            if k == 0:
                target.paste(pipbe[k], (
                    (UNIT_WIGHT + margin) * k + margin, 0 + m_top, (UNIT_WIGHT) * (k + 1) + margin, UNIT_SIZE + m_top))
            else:
                target.paste(pipbe[k], (
                    (UNIT_WIGHT + margin) * k + margin, 0 + m_top, (UNIT_WIGHT + margin) * k + UNIT_WIGHT + margin,
                    UNIT_SIZE + m_top))
        target.save(savepath + '/' + name, quality=100)
        # target = Image.new('L', (TARGET_WIDTH, UNIT_SIZE * 2 + margin * 2), color='white')
        # for k in range(len(pipbe)/2):
        #     if k == 0:
        #         target.paste(pipbe[k], ((UNIT_WIGHT + margin) * k + margin, 0 + m_top, (UNIT_WIGHT) * (k + 1) + margin, UNIT_SIZE + m_top))
        #     else:
        #         target.paste(pipbe[k], ((UNIT_WIGHT + margin) * k + margin, 0 + m_top, (UNIT_WIGHT + margin) * k + UNIT_WIGHT + margin, UNIT_SIZE + m_top))
        # for k in range(len(pipbe)/2, len(pipbe)):
        #     if k == 0:
        #         target.paste(pipbe[k], ((UNIT_WIGHT + margin) * (k - len(pipbe)/2) + margin, 0 + m_top + margin + UNIT_SIZE, (UNIT_WIGHT) * ((k - len(pipbe)/2) + 1) + margin, UNIT_SIZE * 2 + margin + m_top))
        #     else:
        #         target.paste(pipbe[k], ((UNIT_WIGHT + margin) * (k - len(pipbe)/2) + margin, 0 + m_top + margin + UNIT_SIZE, (UNIT_WIGHT + margin) * (k - len(pipbe)/2) + UNIT_WIGHT + margin, UNIT_SIZE * 2 + margin + m_top))
        #
        # target.save(savepath + '/' + name, quality=100)


def catImage3():
    # coofoc = [
    #     '/home/lixiang/lx/pix2pix-pytorch-master/output/cat',
    #     '/home/lixiang/lx/pix2pix-pytorch-master/output/local_warped/760',
    # ]
    coofoc = [
        '/home/lixiang/dataset/photosketch/CUFS/test/sketch',
        '/home/lixiang/lx/pix2pix-pytorch-master/output/CUHK/800',
        '/home/lixiang/lx/pix2pix-pytorch-master/output/CUHK_CLASS/800',
    ]
    path = "/home/xxx/photosketch/files/list_test.txt"
    DIR = coofoc[0]
    savepath = '/home/lixiang/lx/pix2pix-pytorch-master/output/cat'
    len_dir = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    UNIT_SIZE = 500  # 250
    UNIT_WIGHT = 400  # 200
    m_top = 5
    margin = 10
    TARGET_WIDTH = (UNIT_WIGHT + margin) * (len(coofoc)) + margin
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n').split("||")
            name = line[0].split('/')
            print(name, "222")
            pipbe = []
            pip11be = Image.open(coofoc[0] + '/' + name[0] + name[2]).convert('RGB')
            pip12be = Image.open(coofoc[1] + '/' + name[0] + name[2]).convert('RGB')
            pipbe.append(pip11be)
            pipbe.append(pip12be)
            target = Image.new('RGB', (TARGET_WIDTH, UNIT_SIZE + m_top * 2), color='white')
            for k in range(len(pipbe)):
                if k == 0:
                    target.paste(pipbe[k], (
                        (UNIT_WIGHT + margin) * k + margin, 0 + m_top, (UNIT_WIGHT) * (k + 1) + margin,
                        UNIT_SIZE + m_top))
                else:
                    target.paste(pipbe[k], (
                        (UNIT_WIGHT + margin) * k + margin, 0 + m_top, (UNIT_WIGHT + margin) * k + UNIT_WIGHT + margin,
                        UNIT_SIZE + m_top))
            target.save(savepath + '/' + name[0] + name[2], quality=100)
    # for i in range(0, len_dir):
    #     # to_resol = 256
    #     print (i)
    #     pipbe = []
    #     pip11be = Image.open(coofoc[0] + '/' + str(i + 1) + '.jpg').convert('RGB')
    #     pip12be = Image.open(coofoc[1] + '/' + str(i + 1) + '.jpg').convert('RGB')
    #     pipbe.append(pip11be)
    #     pipbe.append(pip12be)
    #     for k in range(2, len(coofoc)):
    #         pip13be = Image.open(coofoc[k] + '/' + str(i + 1) + '.jpg').convert('RGB')
    #         pipbe.append(pip13be)
    #     # pip16be = Image.open(coofoc[6] + '/' + str(i) + '_interpolation.jpg').convert('RGB')
    #     target = Image.new('RGB', (TARGET_WIDTH, UNIT_SIZE * 2 + margin * 2), color='white')
    #     for k in range(len(pipbe) / 2 + 1):
    #         if k == 0:
    #             target.paste(pipbe[k], (
    #             (UNIT_WIGHT + margin) * k + margin, 0 + m_top, (UNIT_WIGHT) * (k + 1) + margin, UNIT_SIZE + m_top))
    #         else:
    #             target.paste(pipbe[k], (
    #             (UNIT_WIGHT + margin) * k + margin, 0 + m_top, (UNIT_WIGHT + margin) * k + UNIT_WIGHT + margin,
    #             UNIT_SIZE + m_top))
    #     for k in range(len(pipbe) / 2 + 1, len(pipbe) + 1):
    #
    #         if k == len(pipbe) / 2 + 1:
    #             continue
    #             # target.paste(pipbe[k], (
    #             # (UNIT_WIGHT + margin) * (k - len(pipbe) / 2) + margin, 0 + m_top + margin + UNIT_SIZE,
    #             # (UNIT_WIGHT) * ((k - len(pipbe) / 2) + 1) + margin, UNIT_SIZE * 2 + margin + m_top))
    #         else:
    #             pipbe[k - 1] = pipbe[k - 1].resize((256, 256))
    #             target.paste(pipbe[k-1], (
    #             (UNIT_WIGHT + margin) * (k-1 - len(pipbe) / 2) + margin, 0 + m_top + margin + UNIT_SIZE,
    #             (UNIT_WIGHT + margin) * (k-1 - len(pipbe) / 2) + UNIT_WIGHT + margin, UNIT_SIZE * 2 + margin + m_top))
    #
    #     target.save(savepath + '/' + str(i) + '.jpg', quality=100)


def catImage4():
    # coofoc = [
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_apd/cyclegan/photo',
    #     # '/home/lixiang/lx/SAND-pytorvh-master/output_apd/apdrawing/test_photo',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_apd/cyclegan/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_apd/pix2pix/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_apd/pix2pix_parsing/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_apd/apdrawing/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_apd/apdrawing2/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_apd/cyclegan/sketch',
    #     # '/home/lixiang/lx/SAND-pytorvh-master/output_apd/apdrawing/test_parsing',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_apd/spade/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_apd/eSPADE_Lgeo/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_apd/eSPADE_Lgeo_Lidn/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_apd/eSPADE_Lgeo_Lidn_2/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_apd/eSPADE_Lgeo_Lidn_parsing/x',
    # ]
    # coofoc = [
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE/photo',
    #     # '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE/vipsl_photo',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/cyclegan/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/pix2pix/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/pix2pix_parsing/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/pix2pix_parsing_gmsdx/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE_noLsem/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/eSPADE_noLsem/x',
    #     # '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE/vipsl_parsing',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE/sketch',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE_enfeature/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE_enfeature_gmsdx/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE_en_gmsdx_weight3.0_70/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE_en_gmsdx_yid/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE_par0.4/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/eSPADE_en_gmsdx_par0.4/x',
    # ]
    # coofoc = [
    #     # '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE/photo',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE/cele_photo',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE_noLsem/cele',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE/cele',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE_par0.4/cele',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/eSPADE_noLsem/cele',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE_enfeature/cele',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/eSPADE_Lgeo/cele',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE/cele_parsing',
    #     # '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE/sketch',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE_enfeature_gmsdx/cele',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/eSPADE_Lgeo_par0.4/cele',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/eSPADE_Lgeo_Dis/cele',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE_en_gmsdx_weight3.0_70/cele',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/eSPADE_Lgeo_Lidn/cele',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE_en_gmsdx_yid/cele',
    # ]
    # coofoc = [
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE/vipsl_photo',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/pix2pix_parsing/vipsl',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE_noLsem/vipsl',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/eSPADE_noLsem/vipsl',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE/vipsl_parsing',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE_enfeature/vipsl',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE_enfeature_gmsdx/vipsl',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE_en_gmsdx_weight3.0_70/vipsl',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE_en_gmsdx_yid/vipsl',
    # ]
    # coofoc = [
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE/cele_photo',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/pix2pix_parsing/cele',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE_noLsem/cele',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/eSPADE_noLsem/cele',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE/cele_parsing',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE_enfeature/cele',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE_enfeature_gmsdx/cele',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE_en_gmsdx_weight3.0_70/cele',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE_en_gmsdx_yid/cele',
    # ]
    # coofoc = [
    #     # '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE_Lsem/photo',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE_Lsem/cele_photo',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/eSPADE/cele',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/eSPADE_Lsem/cele',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/eSPADE_Lgeo/cele',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/eSPADE_Lsem_Lgeo_0.4/cele',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/eSPADE_Lsem_Lgeo_2/cele',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/eSPADE_Lsem_Lgeo_3/cele',
    #     # '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE_Lsem/sketch',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE_Lsem/cele_parsing',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/eSPADE_Lsem_Lgeo/cele',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/eSPADE_Lgeo_Dis/cele',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/eSPADE_Lsem_Lgeo_3_Dis/cele',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/eSPADE_Lsem_Lgeo_Dis/cele',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/eSPADE_Lgeo_Lidn/cele',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/eSPADE_Lsem_Lgeo_3_Lidn/cele',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/eSPADE_Lsem_Lgeo_Lidn/cele',
    # ]
    # coofoc = [
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE_Lsem/photo',
    #     # '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE_Lsem/cele_photo',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_final/pix2pix/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_final/SPADE/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_final/eSPADE/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE_Lsem/sketch',
    #     # '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE_Lsem/cele_parsing',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_final/eSPADE_Lgeo/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_final/eSPADE_Lsem_Lgeo/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_final/eSPADE_Lsem_Lgeo_Dis/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_final/eSPADE_Lsem_Lgeo_Lidn_Dis_1/x',
    # ]
    coofoc = [
        '/home/lixiang/lx/SAND-pytorvh-master/output_apd/cyclegan/photo',
        '/home/lixiang/lx/SAND-pytorvh-master/output_apd/eSPADE_Lgeo_Lsem/x',
        '/home/lixiang/lx/SAND-pytorvh-master/output_apd/eSPADE_Lgeo_Lsem_Lidn_0.5/x',
        '/home/lixiang/lx/SAND-pytorvh-master/output_apd/eSPADE_Lgeo_Lsem_Lidn_1/x',
        '/home/lixiang/lx/SAND-pytorvh-master/output_apd/cyclegan/sketch',
        '/home/lixiang/lx/SAND-pytorvh-master/output_apd/eSPADE_Lgeo_Lsem_Lidn_2/x',
        '/home/lixiang/lx/SAND-pytorvh-master/output_apd/eSPADE_Lgeo_Lsem_Lidn_2.5/x',
        '/home/lixiang/lx/SAND-pytorvh-master/output_apd/eSPADE_Lgeo_Lsem_Lidn_3/x',
    ]
    DIR = coofoc[0]
    savepath = '/home/lixiang/lx/SAND-pytorvh-master/output_apd/cat_Lidn'
    len_dir = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    UNIT_SIZE = 512  # 250
    UNIT_WIGHT = 512  # 200
    m_top = 5
    margin = 10
    TARGET_WIDTH = (UNIT_WIGHT + margin) * (len(coofoc) // 2) + margin
    TARGET_WIDTH = int(TARGET_WIDTH)
    for i in range(0, len_dir):
        # to_resol = 256
        print(i)
        pipbe = []
        pip11be = Image.open(coofoc[0] + '/' + str(i + 1) + '.jpg').convert('RGB')
        pip12be = Image.open(coofoc[1] + '/' + str(i + 1) + '.jpg').convert('L')
        pipbe.append(pip11be)
        pipbe.append(pip12be)
        for k in range(2, len(coofoc)):
            pip13be = Image.open(coofoc[k] + '/' + str(i + 1) + '.jpg').convert('RGB')
            pipbe.append(pip13be)
        # pip16be = Image.open(coofoc[6] + '/' + str(i) + '_interpolation.jpg').convert('RGB')
        target = Image.new('RGB', (TARGET_WIDTH, UNIT_SIZE * 2 + margin * 2), color='white')
        for k in range(len(pipbe) // 2):
            if k == 0:
                target.paste(pipbe[k], (
                    (UNIT_WIGHT + margin) * k + margin, 0 + m_top, (UNIT_WIGHT) * (k + 1) + margin, UNIT_SIZE + m_top))
            else:
                target.paste(pipbe[k], (
                    (UNIT_WIGHT + margin) * k + margin, 0 + m_top, (UNIT_WIGHT + margin) * k + UNIT_WIGHT + margin,
                    UNIT_SIZE + m_top))
        for k in range(len(pipbe) // 2, len(pipbe)):
            if k == 0:
                target.paste(pipbe[k], (
                    (UNIT_WIGHT + margin) * (k - len(pipbe) // 2) + margin, 0 + m_top + margin + UNIT_SIZE,
                    (UNIT_WIGHT) * ((k - len(pipbe) // 2) + 1) + margin, UNIT_SIZE * 2 + margin + m_top))
            else:
                target.paste(pipbe[k], (
                    (UNIT_WIGHT + margin) * (k - len(pipbe) // 2) + margin, 0 + m_top + margin + UNIT_SIZE,
                    (UNIT_WIGHT + margin) * (k - len(pipbe) // 2) + UNIT_WIGHT + margin,
                    UNIT_SIZE * 2 + margin + m_top))

        target.save(savepath + '/' + str(i + 1) + '.jpg', quality=100)


def catImage5():
    # cufs
    # 消融实验
    # coofoc = [
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE_Lsem/photo',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_con/pix2pix/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_final/SPADE/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_final/eSPADE/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_final/eSPADE_Lgeo/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_final/eSPADE_Lsem_Lgeo/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_final/eSPADE_Lsem_Lgeo_Dis/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE_Lsem/sketch',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE_Lsem/photo_parsing',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE_Lsem/sketch_parsing',
    # ]
    # 对比实验
    # coofoc = [
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE_Lsem/photo',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_con/NST-based/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_con/cyclegan/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_con/pix2pix/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_con/pGAN/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_con/Semi-supervised/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_con/MDAL/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_con/KnowledgeTransfer-CUHK/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_con/SPADE_baseline/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_final/eSPADE_Lsem_Lgeo_Dis/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE_Lsem/sketch',
    # ]
    # 鲁棒性实验
    # coofoc = [
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE_Lsem/photo',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_final/eSPADE_Lgeo_Lsem_Dis_y_scale/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_final/eSPADE_Lgeo_Lsem_Dis_y_scale/y',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_final/eSPADE_Lgeo_Lsem_Dis_y_width/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_final/eSPADE_Lgeo_Lsem_Dis_y_width/y',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_final/eSPADE_Lgeo_Lsem_Dis_yy_scale/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_final/eSPADE_Lgeo_Lsem_Dis_yy_scale/y',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_final/eSPADE_Lgeo_Lsem_Dis_yy_width/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_final/eSPADE_Lgeo_Lsem_Dis_yy_width/y',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output/SPADE_Lsem/sketch',
    # ]
    # 外部实验
    coofoc = [
        '/home/lixiang/lx/SAND-pytorvh-master/output_final/SPADE/cele_photo',
        '/home/lixiang/lx/SAND-pytorvh-master/output_final/SPADE/cele_parsing',
        '/home/lixiang/lx/SAND-pytorvh-master/output_con/NST-based/cele',
        '/home/lixiang/lx/SAND-pytorvh-master/output_con/cyclegan/cele',
        '/home/lixiang/lx/SAND-pytorvh-master/output_con/pix2pix/cele',
        '/home/lixiang/lx/SAND-pytorvh-master/output_con/Semi-supervised/cele',
        '/home/lixiang/lx/SAND-pytorvh-master/output_con/SPADE_baseline/cele',
        '/home/lixiang/lx/SAND-pytorvh-master/output_con/SCA-GAN/cele',
        '/home/lixiang/lx/SAND-pytorvh-master/output_final/eSPADE_Lsem_Lgeo_Dis/cele',
    ]
    DIR = coofoc[0]
    savepath = '/home/lixiang/lx/SAND-pytorvh-master/output_con/cat_cele/cat_x_3'
    len_dir = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    UNIT_SIZE = 250  # 250
    UNIT_WIGHT = 200  # 200
    w = 4
    h = 4
    TARGET_WIDTH = (UNIT_WIGHT + 4) * len(coofoc)
    TARGET_WIDTH = int(TARGET_WIDTH)
    # #1
    # x1 = 105
    # y1 = 60
    # x2 = 82
    # y2 = 170
    # 2
    # l_eye
    # x1 = 60
    # y1= 110
    # x2 = 82
    # y2 = 170
    # #3
    x1 = 105
    y1 = 60
    x2 = 60
    y2 = 110
    lenth = 35
    for i in range(0, len_dir):
        print(i)
        pipbe = []
        pip11be = Image.open(coofoc[0] + '/' + str(i + 1) + '.jpg').convert('RGB')
        pip12be = Image.open(coofoc[1] + '/' + str(i + 1) + '.jpg').convert('RGB')
        pipbe.append(pip11be)
        pipbe.append(pip12be)
        for k in range(2, len(coofoc)):
            pip13be = Image.open(coofoc[k] + '/' + str(i + 1) + '.jpg').convert('RGB')
            pipbe.append(pip13be)
        # pip16be = Image.open(coofoc[6] + '/' + str(i) + '_interpolation.jpg').convert('RGB')
        target = Image.new('RGB', (TARGET_WIDTH, (UNIT_SIZE + 98 + 4)), color='#F5F5F5')
        for k in range(0, len(pipbe)):
            target.paste(pipbe[k], ((UNIT_WIGHT + w) * k, 0, (UNIT_WIGHT + w) * k + UNIT_WIGHT, UNIT_SIZE))
            draw = ImageDraw.Draw(target)
            draw.line([(k * (UNIT_WIGHT + w) + x1, y1), (k * (UNIT_WIGHT + w) + x1 + lenth, y1),
                       (k * (UNIT_WIGHT + w) + x1 + lenth, y1 + lenth), (k * (UNIT_WIGHT + w) + x1, y1 + lenth),
                       (k * (UNIT_WIGHT + w) + x1, y1)], width=4,
                      fill="red")
            img_detail1 = pipbe[k].crop((x1, y1, x1 + lenth, y1 + lenth))
            # print(k * UNIT_WIGHT + x1, y1, k* UNIT_WIGHT + x1 + lenth, y1 + lenth)
            img_detail1 = img_detail1.resize((98, 98))
            target.paste(img_detail1,
                         (k * (UNIT_WIGHT + w), UNIT_SIZE + h, k * (UNIT_WIGHT + w) + 98, UNIT_SIZE + 98 + h))
            # print(np.array(img_detail1))
            draw = ImageDraw.Draw(target)
            draw.line([(k * (UNIT_WIGHT + w) + x2, y2), (k * (UNIT_WIGHT + w) + x2 + lenth, y2),
                       (k * (UNIT_WIGHT + w) + x2 + lenth, y2 + lenth), (k * (UNIT_WIGHT + w) + x2, y2 + lenth),
                       (k * (UNIT_WIGHT + w) + x2, y2)], width=4,
                      fill="green")
            img_detail2 = pipbe[k].crop((x2, y2, x2 + lenth, y2 + lenth))
            img_detail2 = img_detail2.resize((98, 98))
            target.paste(img_detail2, (
                k * (UNIT_WIGHT + w) + 98 + 4, UNIT_SIZE + h, k * (UNIT_WIGHT + w) + 2 * 98 + 4, UNIT_SIZE + 98 + h))

        target.save(savepath + '/' + str(i + 1) + '.jpg', quality=100)


def catImage6():
    # 外部实验
    coofoc = [
        '/home/lixiang/lx/SAND-pytorvh-master/output_final/SPADE/test_200_photo',
        '/home/lixiang/lx/SAND-pytorvh-master/output_final/SPADE/test_200_parsing',
        '/home/lixiang/lx/SAND-pytorvh-master/output_con/NST-based/test_200',
        '/home/lixiang/lx/SAND-pytorvh-master/output_con/cyclegan/test_200',
        '/home/lixiang/lx/SAND-pytorvh-master/output_con/pix2pix/test_200',
        '/home/lixiang/lx/SAND-pytorvh-master/output_con/Semi-supervised/test_200',
        '/home/lixiang/lx/SAND-pytorvh-master/output_con/SPADE_baseline/test_200',
        '/home/lixiang/lx/SAND-pytorvh-master/output_con/SCA-GAN/test_200',
        '/home/lixiang/lx/SAND-pytorvh-master/output_final/eSPADE_Lsem_Lgeo_Dis/test_200',
    ]
    DIR = coofoc[0]
    savepath = '/home/lixiang/lx/SAND-pytorvh-master/output_con/cat_test_200/cat_x_1'
    len_dir = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    UNIT_SIZE = 200  # 250
    UNIT_WIGHT = 200  # 200
    w = 4
    h = 4
    TARGET_WIDTH = (UNIT_WIGHT + w) * len(coofoc)
    TARGET_WIDTH = int(TARGET_WIDTH)
    # #1
    x1 = 105
    y1 = 15
    x2 = 82
    y2 = 135
    # 2
    # l_eye
    # x1 = 60
    # y1= 83
    # x2 = 82
    # y2 = 135
    # #3
    # x1 = 105
    # y1 = 15
    # x2 = 60
    # y2 = 83
    lenth = 35
    for i in range(0, len_dir):
        print(i)
        pipbe = []
        pip11be = Image.open(coofoc[0] + '/' + str(i + 1) + '.jpg').convert('RGB')
        pip12be = Image.open(coofoc[1] + '/' + str(i + 1) + '.jpg').convert('RGB')
        pipbe.append(pip11be)
        pipbe.append(pip12be)
        for k in range(2, len(coofoc)):
            pip13be = Image.open(coofoc[k] + '/' + str(i + 1) + '.jpg').convert('RGB')
            pipbe.append(pip13be)
        # pip16be = Image.open(coofoc[6] + '/' + str(i) + '_interpolation.jpg').convert('RGB')
        target = Image.new('RGB', (TARGET_WIDTH, (UNIT_SIZE + 98 + 4)), color='#F5F5F5')
        for k in range(0, len(pipbe)):
            target.paste(pipbe[k], ((UNIT_WIGHT + w) * k, 0, (UNIT_WIGHT + w) * k + UNIT_WIGHT, UNIT_SIZE))
            draw = ImageDraw.Draw(target)
            draw.line([(k * (UNIT_WIGHT + w) + x1, y1), (k * (UNIT_WIGHT + w) + x1 + lenth, y1),
                       (k * (UNIT_WIGHT + w) + x1 + lenth, y1 + lenth), (k * (UNIT_WIGHT + w) + x1, y1 + lenth),
                       (k * (UNIT_WIGHT + w) + x1, y1)], width=4,
                      fill="red")
            img_detail1 = pipbe[k].crop((x1, y1, x1 + lenth, y1 + lenth))
            # print(k * UNIT_WIGHT + x1, y1, k* UNIT_WIGHT + x1 + lenth, y1 + lenth)
            img_detail1 = img_detail1.resize((98, 98))
            target.paste(img_detail1,
                         (k * (UNIT_WIGHT + w), UNIT_SIZE + h, k * (UNIT_WIGHT + w) + 98, UNIT_SIZE + 98 + h))
            # print(np.array(img_detail1))
            draw = ImageDraw.Draw(target)
            draw.line([(k * (UNIT_WIGHT + w) + x2, y2), (k * (UNIT_WIGHT + w) + x2 + lenth, y2),
                       (k * (UNIT_WIGHT + w) + x2 + lenth, y2 + lenth), (k * (UNIT_WIGHT + w) + x2, y2 + lenth),
                       (k * (UNIT_WIGHT + w) + x2, y2)], width=4,
                      fill="green")
            img_detail2 = pipbe[k].crop((x2, y2, x2 + lenth, y2 + lenth))
            img_detail2 = img_detail2.resize((98, 98))
            target.paste(img_detail2, (
                k * (UNIT_WIGHT + w) + 98 + 4, UNIT_SIZE + h, k * (UNIT_WIGHT + w) + 2 * 98 + 4, UNIT_SIZE + 98 + h))

        target.save(savepath + '/' + str(i + 1) + '.jpg', quality=100)


def catImage7():
    # coofoc = [
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_apd/cyclegan/photo',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_apd/cyclegan/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_apd/pix2pix/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_apd/pix2pix_parsing/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_apd/apdrawing/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_apd/apdrawing2/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_apd/eSPADE_Lgeo_Lsem_Lidn_2/x',
    #     '/home/lixiang/lx/SAND-pytorvh-master/output_apd/cyclegan/sketch',
    # ]
    coofoc = [
        '/home/lixiang/lx/SAND-pytorvh-master/output_apd/SPADE/test_photo',
        '/home/lixiang/lx/SAND-pytorvh-master/output_apd/cyclegan/test',
        '/home/lixiang/lx/SAND-pytorvh-master/output_apd/pix2pix/test',
        '/home/lixiang/lx/SAND-pytorvh-master/output_apd/apdrawing/test',
        '/home/lixiang/lx/SAND-pytorvh-master/output_apd/apdrawing2/test',
        '/home/lixiang/lx/SAND-pytorvh-master/output_apd/eSPADE_Lgeo_Lsem_Lidn_2/test',
        '/home/lixiang/lx/SAND-pytorvh-master/output_apd/SPADE/test_parsing',
    ]
    DIR = coofoc[0]
    savepath = '/home/lixiang/lx/SAND-pytorvh-master/output_apd/cat_test/cat_test_5'
    len_dir = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    UNIT_SIZE = 512  # 250
    UNIT_WIGHT = 512  # 200
    w = 8
    TARGET_WIDTH = (UNIT_WIGHT + w) * len(coofoc)
    TARGET_WIDTH = int(TARGET_WIDTH)
    # #1
    # x1 = 210
    # y1 = 100
    # x2 = 220
    # y2 = 340
    # 2
    # l_eye
    # x1 = 160
    # y1 = 210
    # x2 = 220
    # y2 = 340
    # #3
    # x1 = 210
    # y1 = 100
    # x2 = 160
    # y2 = 210
    # 4 眼睛、胡子
    # x1 = 160
    # y1= 210
    # x2 = 220
    # y2 = 290
    # x1 = 180
    # y1= 50
    # x2 = 160
    # y2 = 210
    # 5 眼睛、衣服
    # x1 = 160
    # y1 = 210
    # x2 = 400
    # y2 = 400
    x1 = 160
    y1 = 210
    x2 = 240
    y2 = 450
    lenth = 70
    for i in range(0, len_dir):
        print(i)
        pipbe = []
        pip11be = Image.open(coofoc[0] + '/' + str(i + 1) + '.jpg').convert('RGB')
        pip12be = Image.open(coofoc[1] + '/' + str(i + 1) + '.jpg').convert('RGB')
        pipbe.append(pip11be)
        pipbe.append(pip12be)
        for k in range(2, len(coofoc)):
            pip13be = Image.open(coofoc[k] + '/' + str(i + 1) + '.jpg').convert('RGB')
            pipbe.append(pip13be)
        # pip16be = Image.open(coofoc[6] + '/' + str(i) + '_interpolation.jpg').convert('RGB')
        target = Image.new('RGB', (TARGET_WIDTH, (UNIT_SIZE + 252 + 8)), color='#F5F5F5')
        for k in range(0, len(pipbe)):
            target.paste(pipbe[k], ((UNIT_WIGHT + w) * k, 0, (UNIT_WIGHT + w) * k + UNIT_WIGHT, UNIT_SIZE))
            draw = ImageDraw.Draw(target)
            draw.line([(k * (UNIT_WIGHT + w) + x1, y1), (k * (UNIT_WIGHT + w) + x1 + lenth, y1),
                       (k * (UNIT_WIGHT + w) + x1 + lenth, y1 + lenth), (k * (UNIT_WIGHT + w) + x1, y1 + lenth),
                       (k * (UNIT_WIGHT + w) + x1, y1)], width=4,
                      fill="red")

            img_detail1 = pipbe[k].crop((x1, y1, x1 + lenth, y1 + lenth))
            # print(k * UNIT_WIGHT + x1, y1, k* UNIT_WIGHT + x1 + lenth, y1 + lenth)
            img_detail1 = img_detail1.resize((252, 252))
            target.paste(img_detail1,
                         (k * (UNIT_WIGHT + w), UNIT_SIZE + 8, k * (UNIT_WIGHT + w) + 252, UNIT_SIZE + 252 + 8))
            # print(np.array(img_detail1))
            draw = ImageDraw.Draw(target)
            draw.line([(k * (UNIT_WIGHT + w) + x2, y2), (k * (UNIT_WIGHT + w) + x2 + lenth, y2),
                       (k * (UNIT_WIGHT + w) + x2 + lenth, y2 + lenth), (k * (UNIT_WIGHT + w) + x2, y2 + lenth),
                       (k * (UNIT_WIGHT + w) + x2, y2)], width=4,
                      fill="green")
            img_detail2 = pipbe[k].crop((x2, y2, x2 + lenth, y2 + lenth))
            img_detail2 = img_detail2.resize((252, 252))
            target.paste(img_detail2,
                         (k * (UNIT_WIGHT + w) + 8 + 252, 512 + 8, k * (UNIT_WIGHT + w) + 8 + 2 * 252, 512 + 252 + 8))

        target.save(savepath + '/' + str(i + 1) + '.jpg', quality=100)


if __name__ == '__main__':
    catImage4()
    # cropImage()
