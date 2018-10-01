# coding:utf-8

import pickle
import os
import pandas as pd
import numpy as np
import time


# def pre_process_data(data_temp):
#     ddata = np.zeros((1, 201)) # build 2*100 new array
#
#     # from DataFrame to array
#     data_temp = np.array(data_temp)
#     # acquire the index of the sorted values
#     sorted_index = np.argsort(data_temp[0, 1:])
#
#     ddata[0, 0] = data_temp[0, 1]
#     for i in range(1, 101):  # i=1~100
#         ddata[0, i] = sorted_index[40183 - i]+1
#         ddata[0, i + 100] = data_temp[0, sorted_index[40183 - i]+1]
#
#     return ddata
#
#
# # 在程序所在目录下创建一个test.data文件
# start_time = time.time()
# pklfile = open("./Dataset/test.pkl", "ab")
# rootDir = r'C:\Users\King\Desktop\tensorflow\torch_model_2\Dataset\Traindataset'
#
# for lists in os.listdir(rootDir):
#      # print(lists)
#      str1 = os.path.join(rootDir, lists)
#      print("present dir:" + lists)
#      for root, dirs, files in os.walk(str1):
#         # print(files)
#         for file in files:
#             str2 = os.path.join(str1, file)
#
#             print("  ----start to read csv file:" + file)
#             data_temp = pd.read_csv(str2, header=None)
#
#             data = pre_process_data(data_temp)
#             pickle.dump(data, pklfile)
#             print("      Data has been saved into pkl dataset")
#
# print("ALL over")
# pklfile.close()
# print((time.time()-start_time))


def pre_process_data(data_temp):
    ddata = np.zeros((2, 100))

    # from DataFrame to array
    data_temp = np.array(data_temp)
    # acquire the index of the sorted values
    sorted_index = np.argsort(data_temp[0, 1:])

    label = data_temp[0, 0]
    for i in range(0, 100):  # i=0~99
        ddata[0, i] = sorted_index[40183 - i]+1
        ddata[1, i] = data_temp[0, sorted_index[40183 - i]+1]

    list_temp = [ddata, label]
    return list_temp


# 在程序所在目录下创建一个test.data文件
rootDir = r'C:\Users\King\Desktop\tensorflow\torch_model_2\Dataset\Traindataset'
Total_data_list = []
start_time = time.time()
M = 0
N = 0

for lists in os.listdir(rootDir):
     # print(lists)
     str1 = os.path.join(rootDir, lists)
     print("present dir:" + lists)
     N = N+1
     for root, dirs, files in os.walk(str1):
        # print(files)
        for file in files:
            str2 = os.path.join(str1, file)
            M = M+1
            print("  ----Dir Id:" + str(N) + ", File Id:" + str(M) + ", start to read csv file:" + file)
            data_temp = pd.read_csv(str2, header=None)
            data = pre_process_data(data_temp)
            Total_data_list.append(data)
            print("      Data has been saved into pkl dataset")

print("A1: Data_abstraction process is over.")
print("A1: Time consumption(s):" + str(time.time()-start_time))

start_time = time.time()
pklfile = open("./Dataset/datafortraining.pkl", "ab")
pickle.dump(Total_data_list, pklfile)
print("A2: Data saving process is over.")
print("A2: Time consumption(s):" + str(time.time()-start_time))
pklfile.close()
