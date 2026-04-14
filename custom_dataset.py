import os
import pandas as pd
from torchvision.io import decode_image
from torch.utils.data import Dataset
from pathlib import Path

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, num_soundfonts, soundfont_map, transform=None):
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.num_soundfonts = num_soundfonts
        self.sf_map = soundfont_map
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)*self.num_soundfonts

    def __getitem__(self, idx):
        # print(f"img idx = {idx}")
        
        img_num = int(idx / self.num_soundfonts) + 1
        sf_idx = idx % self.num_soundfonts

        img_path = f"{self.img_dir}/spect_{img_num:03}_{self.sf_map[sf_idx]}.png"
        #print(img_path)
        image = decode_image(img_path)
        label = self.img_labels.iloc[img_num - 1, 1]
        if self.transform:
            image = self.transform(image)
        return image, label