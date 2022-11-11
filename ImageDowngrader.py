import cv2
import os

class ImageDowngrader:

    def __init__(self, src_path, des_path, factor):

        if os.path.exists(src_path):
            self.src_path = src_path
        else:
            raise Exception("src_path: does not exist")
        
        if not os.path.exists(des_path):
            os.mkdir(des_path)
        
        self.des_path = des_path
        self.factor = factor

    def reduceQuality(self):
        for img in os.listdir(self.src_path):

            curr_image = cv2.imread(self.src_path + "/" + img)

            width = int(curr_image.shape[1] * self.factor / 100)
            height = int(curr_image.shape[0] * self.factor / 100)
            dim = (width, height)

            downsized = cv2.resize(curr_image, dim)

            width *= int(100 / self.factor)
            height *= int(100 / self.factor)
            dim = (width, height)

            upsized = cv2.resize(downsized, dim)

            cv2.imwrite(self.des_path + "/" + img, upsized)

# obj = ImageDowngrader("raw_data/high_res", "raw_data/low_res", 50)
# obj.reduceQuality()