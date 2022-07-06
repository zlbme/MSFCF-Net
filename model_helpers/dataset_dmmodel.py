"""
Dataset class for dual-modal EIT image reconstruction
@author: LIU Zhe
"""
from torch.utils.data import Dataset, DataLoader
import numpy as np
from matplotlib import pyplot as plt


class DatasetDualVanilla(Dataset):
    """
    Vanilla version of dataset for dual-modal imaging 
    """
    def __init__(self, v_path, msk_path, img_path):

        super().__init__()

        v_np = np.load(v_path).astype(np.float32)
        msk_np = np.load(msk_path).astype(np.float32)
        img_np = np.load(img_path).astype(np.float32)

        dataset = []
        for i in range(len(v_np)):
            dataset.append((v_np[i], msk_np[i], img_np[i]))

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        v, msk, img = self.dataset[index]
        # print('The type of voltages is {}, and its shape is {}.'.format(type(v), v.shape))
        # print('The type of mask is {}, and its shape is {}.'.format(type(msk), msk.shape))

        return v, msk, img




