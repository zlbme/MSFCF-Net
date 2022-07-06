"""
Dataset classes for dual modal EIT image reconstruction
"""
from torch.utils.data import Dataset, DataLoader
import numpy as np
from matplotlib import pyplot as plt


class DatasetDualVanilla(Dataset):
    """
    Vanilla version of dual modal data set
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


if __name__ == '__main__':

    proj_path = "D:/PhD/My Projects/EIT/EIT Image Reconstruction/dmEIT2D - MSFCFNet/"

    # v_path = proj_path + 'dataSimu/obj1234_TrainSet/v_train.npy'
    # msk_path = proj_path + 'dataSimu/obj1234_TrainSet/msk_train.npy'
    # img_path = proj_path + 'dataSimu/obj1234_TrainSet/img_train.npy'

    # v_path = proj_path + 'dataSimu/obj1/obj1_psd/vClean_train.npy'
    # msk_path = proj_path + 'dataSimu/obj1/obj1_psd/mskClean_train.npy'
    # img_path = proj_path + 'dataSimu/obj1/obj1_psd/img_train.npy'

    v_path = proj_path + 'dataSimu/obj1234_TestSet/vClean_test.npy'
    msk_path = proj_path + 'dataSimu/obj1234_TestSet/mskClean_test.npy'
    img_path = proj_path + 'dataSimu/obj1234_TestSet/img_test.npy'

    data_set = DatasetDualVanilla(v_path, msk_path, img_path)

    batch_size = 100
    train_loader = DataLoader(data_set, batch_size=batch_size)

    for batch_v, batch_msk, batch_img in train_loader:

        print('--------------------------------------------')
        print(batch_v.shape)
        print(batch_msk.shape)
        print(batch_img.shape)
        print('--------------------------------------------')

        slice_num = 11
        msk = batch_msk.numpy()[slice_num, 0, ...]
        img = batch_img.numpy()[slice_num, 0, ...]

        print('--------------------------------------------')
        print('The shape of the selected mask is {}.'.format(msk.shape))
        print('The shape of the selected image is {}.'.format(img.shape))
        print('--------------------------------------------')

        # Display data
        plt.subplot(121)
        plt.imshow(msk, vmin=0, vmax=1, interpolation='nearest')
        plt.title('Mask Image')
        plt.colorbar(shrink=0.5)

        plt.subplot(122)
        plt.imshow(img, vmin=0, vmax=1, cmap='Purples', interpolation='nearest')
        plt.title('EIT Image')
        plt.colorbar(shrink=0.5)
        plt.show()
        break















































