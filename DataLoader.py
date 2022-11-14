import os
import cv2
import numpy as np

class DataLoader():

    def __init__(self, src_path, split):
        self.src_path = src_path
        self.split = split
        self.width = 0
        self.height = 0
        self.channels = 0

    def loadData(self):

        if not os.path.exists(self.src_path):
            raise Exception("src_path: does not exist")
        
        high_res = []
        low_res = []
        curr_img = None

        for img in sorted(os.listdir(self.src_path + "/" + "high_res")):
            curr_img = cv2.imread(self.src_path + "/" + "high_res/" + img, 1)
            curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB)
            curr_img = curr_img.astype('float32') / 255.0
            high_res.append(curr_img)

        for img in sorted(os.listdir(self.src_path + "/" + "low_res")):
            curr_img = cv2.imread(self.src_path + "/" + "low_res/" + img, 1)
            curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB)
            curr_img = curr_img.astype('float32') / 255.0
            low_res.append(curr_img)

        self.width = curr_img.shape[0]
        self.height = curr_img.shape[1]
        self.channels = curr_img.shape[2]


        return self.splitData(np.array(low_res), np.array(high_res))

    def splitData(self, low_res, high_res):

        if not all(0 <= i <= 1 for i in self.split):
            raise Exception("split: all values must be between 0 and 1")

        low_res_train, low_res_remaining  = np.split(low_res, [int(len(low_res)*self.split[0])])
        low_res_val, low_res_test = np.split(low_res_remaining, [int(len(low_res_remaining)*self.split[1])])


        high_res_train, high_res_remaining  = np.split(high_res, [int(len(high_res)*self.split[0])])
        high_res_val, high_res_test = np.split(high_res_remaining, [int(len(high_res_remaining)*self.split[1])])


        return(
            low_res_train,
            low_res_test,
            low_res_val,
            high_res_train,
            high_res_test,
            high_res_val
        )