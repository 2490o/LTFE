# 测试
# 使用小的文件夹进行测试 只有五张图片，能够进行损坏并保存到指定位置
# 使用 train.txt中文件进行测试，四张图片，也能够成功保存到指定位置
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

    def process_images(self):
        """遍历文件夹下所有图像文件，应用破坏效果，并保存结果。"""
        for filename in os.listdir(self.src_dir):
            if filename.endswith('.jpg'):
                img_path = os.path.join(self.src_dir, filename)
                img = Image.open(img_path).convert("RGB")
                corrupted_img = self.corruption(np.array(img))
                
                # 保存处理后的图像
                save_path = os.path.join(self.dest_dir, filename)
                corrupted_img.save(save_path)
                print(f"Processed and saved {filename} to {self.dest_dir}")

    def process_train_txt_images(self):
        """遍历train.txt下的所有图像文件，应用破坏效果，并保存结果。"""
        with open('/data1/ysk/Div_code/day_clear/VOC2007/ImageSets/train.txt', 'r') as f:
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
    src_dir = "./day_clear/VOC2007/JPEGImages"
    # dest_dir = "./corrupt_img"
    dest_dir = "./train_corrupt_img"
    
    # 初始化处理器并处理图像
    processor = ImageCorruptionProcessor(src_dir, dest_dir)
    # processor.process_images()
    processor.process_train_txt_images()# 最后保存的图像名前加了一个cp,表示corrupt
