# 实际使用 环境是gnas
# 将day_clear的train.txt中的文件全部进行损坏，存储到cp_day_clear中  
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
            "phase_scaling"
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

    def process_train_txt_images(self):
        """遍历train.txt下的所有图像文件，应用破坏效果，并保存结果。"""
        with open('/home/zzh/data/diverseWeather/daytime_clear/VOC2007/ImageSets/Main/train.txt', 'r') as f:
            image_files = [line.strip() for line in f]
        for filename in image_files:
            img_path = os.path.join(self.src_dir, filename+'.jpg')
            img = Image.open(img_path).convert("RGB")
            corrupted_img = self.corruption(np.array(img))
            
            # 保存处理后的图像
            save_path = os.path.join(self.dest_dir, 'cp' + filename + '.jpg')
            corrupted_img.save(save_path)
            print(f"Processed and saved {filename} to {self.dest_dir}")



if __name__ == "__main__":
    # 源图像目录和目标目录
    src_dir = "/home/zzh/data/diverseWeather/daytime_clear/VOC2007/JPEGImages"
    dest_dir = "/home/zzh/data/diverseWeather/daytime_clear/VOC2007/cp_train_corrupt_img"
    
    # 初始化处理器并处理图像
    processor = ImageCorruptionProcessor(src_dir, dest_dir)
    processor.process_train_txt_images()# 最后保存的图像名前加了一个cp,表示corrupt
