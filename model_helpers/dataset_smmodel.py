# Author: Zhe Liu
# Date: 2020
from torch.utils.data import Dataset, DataLoader
import numpy as np
from matplotlib import pyplot as plt


class DatasetSingleVanilla(Dataset):
    """
    Vanilla version of single modal model data set
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


if __name__ == '__main__':

    proj_path = "D:/PhD/My Projects/EIT/EIT Image Reconstruction/dmEIT2D - MSFCFNet/"
    v_path = proj_path + 'dataSimu/obj1234_TrainSet/v_train.npy'
    img_path = proj_path + 'dataSimu/obj1234_TrainSet/img_train.npy'

    data_set = DatasetSingleVanilla(v_path, img_path)
    train_loader = DataLoader(data_set, batch_size=50)

    for batch_v, batch_img in train_loader:

        print('--------------------------------------------')
        print(batch_v.shape)
        print(batch_img.shape)
        print('--------------------------------------------')

        slice_num = 3
        img = batch_img.numpy()[slice_num, 0, ...]

        print('--------------------------------------------')
        print(img.shape)
        print('--------------------------------------------')

        # Display
        plt.imshow(img, vmin=0, vmax=1, cmap='Purples', interpolation='nearest')
        plt.title('EIT Image')
        plt.colorbar()
        plt.show()

        break









