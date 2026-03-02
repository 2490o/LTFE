# 测试
# 使用一张图像进行所有的损失变换，将这些结果进行保存，观察是否发生了位移
# 出了一个扭曲变换，都没有位移，扭曲变换位移也可以忽略
# 都不影响继续使用之前的标注信息 能够直接进行处理，进行增广
import os
import numpy as np
from PIL import Image
import random
from ACVCGenerator import ACVCGenerator

class ImageCorruptionProcessor:
    def __init__(self, src_dir, dest_dir):
        self.src_dir = src_dir
        self.dest_dir = dest_dir
        self.acvc = ACVCGenerator()
        self.corruption_func = [
            "defocus_blur",
            "glass_blur",
            "gaussian_blur",
            "motion_blur",
            "speckle_noise",
            "shot_noise",
            "impulse_noise",
            "gaussian_noise",
            "jpeg_compression",
            "pixelate",
            "elastic_transform",
            "brightness",
            "saturate",
            "contrast",
            "high_pass_filter",
            "phase_scaling",

            # "constant_amplitude", # 不选这个，因为他得到的图像丢失的信息太多了，变成了几乎黑图
        ]
        self.n_augs = 1  # 每次选一种破坏效果

        # 创建目标文件夹
        os.makedirs(self.dest_dir, exist_ok=True)

    def apply_corruption(self, img, corruption_type, severity=1):
        """使用指定的破坏类型和强度应用到图像。"""
        return self.acvc.apply_corruption(img, corruption_type, severity)

    def corruption(self, img):
        """随机选择一种破坏效果应用到图像，并返回处理后的图像。"""
        crs = random.sample(self.corruption_func, self.n_augs)
        images = []
        for c in crs:
            severity = random.randint(1, 5)
            aug_img = self.apply_corruption(img, c, severity).convert("RGB")
            images.append(aug_img)
        return images[0]  # 返回一个图像
    
    def corruption_all(self, img):
        # 使用所有的破坏方法进行实验
        images = []
        for corr_fun in self.corruption_func:
            severity = random.randint(1, 5)
            aug_img = self.apply_corruption(img, corr_fun, severity).convert("RGB")
            images.append(aug_img)
        return images # 返回所有的图像


    def process_images(self):
        """遍历所有图像文件，应用破坏效果，并保存结果。"""
        filename = '/data1/ysk/Div_code/day_clear/VOC2007/JPEGImages/000e0252-8523a4a9.jpg'
        img_path = os.path.join(self.src_dir, filename)
        img = Image.open(img_path).convert("RGB")
        all_corrupted_img = self.corruption_all(np.array(img))

        i = 0
        for corrupted_img in all_corrupted_img:
            # 保存处理后的图像
            save_path = os.path.join(self.dest_dir, '000e0252-8523a4a9' + str(i) + '.jpg')
            corrupted_img.save(save_path)
            print(f"Processed and saved {filename} to {self.dest_dir}")
            i = i + 1

if __name__ == "__main__":
    # 源图像目录和目标目录
    src_dir = "./day_clear/VOC2007/JPEGImages"
    dest_dir = "./all_corrupt_img"
    
    # 初始化处理器并处理图像
    processor = ImageCorruptionProcessor(src_dir, dest_dir)
    processor.process_images()
