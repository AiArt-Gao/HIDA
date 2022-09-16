import os
import cv2
from glob import glob

path = '/home/lixiang/lx/GRID-pytorch-master/result/GRID_apd/1.9/orggan/apd/549/x'
path_save = '/home/lixiang/lx/GRID-pytorch-master/output/GRID_apd'
listname = os.listdir(path)
# listname = sorted(glob("{}/*.jpg".format(path)), key=os.path.getctime)
# listname.sort(key= lambda x:int(x[:-4]))
for name in listname:
    print(os.path.join(path, name))
    img = cv2.imread(os.path.join(path, name))
    img = img[:, 1024:1536, :]
    # img = img[:, 1200:1600, :]
    # img=img[:,400:600,:]
    # img=img[25:225,:,:]
    cv2.imwrite(os.path.join(path_save, name), img)
# import scipy.io as sio
# import kornia
# import torch
# import os
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import torch.nn.functional as F
# import cv2
# def get_mat(matpath):
#     facelabel = sio.loadmat(matpath)
#     temp = facelabel['res_label']
#     return temp
# def vis_parsing_maps(parsing_anno,save_path):
#     part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
#                                   [255, 0, 85], [255, 0, 170],
#                                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
#                                   [0, 255, 85], [0, 255, 170],
#                                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
#                                   [0, 85, 255], [0, 170, 255],
#                                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
#                                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
#                                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]
#     parsing_anno = parsing_anno.squeeze(0).cpu().numpy().argmax(0)
#     vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
#     vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
#
#     num_of_class = np.max(vis_parsing_anno)
#     for pi in range(1, num_of_class+1):
#         index = np.where(vis_parsing_anno == pi)
#         vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]
#
#     vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
#     # print(np.unique(vis_parsing_anno))
#     vis_parsing_anno_color = vis_parsing_anno_color
#     cv2.imwrite(save_path[:-4] + '.jpg', vis_parsing_anno_color, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
#
# path="/home/lixiang/dataset/spade/files/list_test.txt"
# path2="/home/lixiang/dataset/spade/files/list_train.txt"
# path1='/home/lixiang/dataset/spade'
# color=torch.zeros((250,200))
# def combine_lr(parsing_anno):
#     parsing = parsing_anno.squeeze(0).cpu().numpy().copy()  # [19,250,200]
#     parsing_anno = parsing.argmax(0)
#     ips = np.unique(parsing_anno)
#     if 3 in ips:
#         # index = np.where(parsing_anno == 3)
#         parsing[2] += parsing[3]
#     if 5 in ips:
#         parsing[4] += parsing[5]
#     if 8 in ips:
#         parsing[7] += parsing[8]
#     parsing = np.delete(parsing, [3, 5, 8], axis=0)
#     parsing = torch.from_numpy(parsing).unsqueeze(0)
#     return parsing
# sum=torch.zeros((606,250,200))
# i=0
# with open(path) as f:
#     for line in f.readlines():
#         print(line)
#         name=line.strip().split("||")
#         img_par=get_mat(os.path.join(path1,name[2]))
#         # img_par = kornia.image_to_tensor(img_par, keepdim=False).float()
#         # vis_parsing_maps(img_par,"/home/lixiang/lx/SAND-pytorvh-master/output_final/eSPADE_Lsem_Lgeo_Dis/1.jpg")
#         # img=combine_lr(img_par)
#         # vis_parsing_maps(img,"/home/lixiang/lx/SAND-pytorvh-master/output_final/eSPADE_Lsem_Lgeo_Dis/2.jpg")
#         sketch_par=get_mat(os.path.join(path1,name[3]))
#         img_par = kornia.image_to_tensor(img_par, keepdim=False).float()
#         sketch_par = kornia.image_to_tensor(sketch_par, keepdim=False).float()
#         img_par=combine_lr(img_par)
#         sketch_par=combine_lr(sketch_par)
#         cos_sim = F.cosine_similarity(img_par, sketch_par, dim=1)
#         cos_sim = torch.abs(cos_sim)
#         cos_sim = torch.squeeze(cos_sim, dim=0)
#         sum[i]=cos_sim
#         i=i+1
# with open(path2) as f1:
#     for line in f1.readlines():
#         print(line)
#         name=line.strip().split("||")
#         img_par=get_mat(os.path.join(path1,name[2]))
#         sketch_par=get_mat(os.path.join(path1,name[3]))
#         img_par = kornia.image_to_tensor(img_par, keepdim=False).float()
#         sketch_par = kornia.image_to_tensor(sketch_par, keepdim=False).float()
#         img_par = combine_lr(img_par)
#         sketch_par = combine_lr(sketch_par)
#         cos_sim = F.cosine_similarity(img_par, sketch_par, dim=1)
#         cos_sim = torch.abs(cos_sim)
#         cos_sim = torch.squeeze(cos_sim, dim=0)
#         sum[i] = cos_sim
#         i = i + 1
# # color=color/606
# # print(torch.unique(color))
# # for i in range(15):
# #     print(torch.unique(color[i]))
# #     sns.set()
# #     ax = sns.heatmap(color[i], cmap="Blues",xticklabels=False,yticklabels=False)
# #     plt.savefig("/home/lixiang/lx/SAND-pytorvh-master/output_final/hotmap/parsing_combine/"+str(i)+'.jpg')
# #     plt.show()
# # sum_color=torch.sum(color,axis=0)
# # sns.set()
# # ax = sns.heatmap(color,cmap="Blues",xticklabels=False,yticklabels=False)
# # plt.savefig("/home/lixiang/lx/SAND-pytorvh-master/output_final/hotmap/parsing_combine/20.jpg")
# # plt.show()
# sum=sum.reshape(-1)
# print(torch.unique(sum))
# plt.hist(sum, bins=50, color='g',weights= [1./ len(sum)] * len(sum))
# plt.semilogy()
# plt.savefig("/home/lixiang/lx/SAND-pytorvh-master/output_final/hotmap/parsing_combine/100.jpg")
# plt.show()
# plt.hist(sum, bins=50, cumulative=True, color='r',weights= [1./ len(sum)] * len(sum))
# plt.savefig("/home/lixiang/lx/SAND-pytorvh-master/output_final/hotmap/parsing_combine/101.jpg")
# plt.semilogy()
# plt.show()
# # print(torch.unique(sum_color))
# # print(sum_color.shape)

# import cv2,os
# path='/home/lixiang/dataset/Zjj/all_photo_align'
# str1='all_photo_align/'
# str2='all_sketch_align/'
# str3='all_parsing_align_14/'
# path1='/home/lixiang/dataset/Zjj/files/list_train.txt'
# with open(path1,'w') as f:
#     listname=os.listdir(path)
#     for name in listname:
#         name1=name.replace('.png','.mat')
#         line=str1+name+"||"+str2+name+"||"+str3+name1+"||"+str3+name1+'\n'
#         f.write(line)
#         print(line)
# f.close()
