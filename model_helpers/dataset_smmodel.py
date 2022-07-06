'''
Dataset class for single-modal deep learning model
@author: LIU Zhe
'''
from torch.utils.data import Dataset, DataLoader
import numpy as np
from matplotlib import pyplot as plt


class DatasetSingleVanilla(Dataset):
    """
    Vanilla version of dataset for single-modal deep learning model
    """
    def __init__(self, v_path, img_path):
        super().__init__()

        v_np = np.load(v_path).astype(np.float32)
        img_np = np.load(img_path).astype(np.float32)

        dataset = []
        for i in range(len(v_np)):
            dataset.append((v_np[i], img_np[i]))

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        v, img = self.dataset[index]
        return v, img


    
    
