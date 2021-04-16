import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class MelanomaDataset(Dataset):
    """Melanoma dataset"""

    def __init__(self, csv_file, root_dir, label_type='target', img_format='dcm', transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            label_type (string): Label type for each task.
                                  * For the hair removal task -> 'hair'
                                  * For the classification task -> 'target'
            img_format (string): Image data type to load.
                                  * dcm -> 'dcm'
                                  * jpg -> 'jpg'
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.label_type = label_type
        self.img_format = img_format
        self.transform = transform
    
    def __len__(self):
        return len(self.df)



    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = '{}/{}.{}'.format(self.root_dir, self.df.iloc[idx]['image_name'], self.img_format)

        if self.img_format == 'jpg':
            img = Image.open(img_path)
            img = np.array(img) / 255
            img = np.float32(img)
        else:
            ds = pydicom.read_file(img_path)
            arr = ds.pixel_array
            arr_scaled = arr / 255
            img = arr_scaled
            img = np.float32(img)
        label = self.df.iloc[idx][self.label_type]

        if self.transform:
            img = self.transform(img)

        data_dict = {'image': img, 'label': label}

        return data_dict




# the following code snippet uses tensorflow.

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import tensorflow as tf
# import tensorflow_io as tfio

    # def __getitem__(self, idx):
    #     if torch.is_tensor(idx):
    #         idx = idx.tolist()

    #     img_path = '{}/{}.{}'.format(self.root_dir, self.df.iloc[idx]['image_name'], self.img_format)

    #     image_bytes = tf.io.read_file(img_path)
    #     image = tfio.image.decode_dicom_image(image_bytes, dtype=tf.uint16)
    #     print(image[0].shape)
    #     print(image[0][0,0,:])
    #     # print(image[0][0][0][0], image.shape)
    #     img = image[0] / 255
    #     img = np.float32(img)
    #     label = self.df.iloc[idx][self.label_type]

    #     if self.transform:
    #         img = self.transform(img)

    #     data_dict = {'image': img, 'label': label}

    #     return data_dict
