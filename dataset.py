import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
import transforms as T
from glob import glob


class ProstateDataset(Dataset):
    ''' Prostate dataset
    '''

    def __init__(self, anno, root='/work/nvme/beej/ylin20/jhu_data/all_data', 
                 data_folder='nips_downsample_data', sos_folder='nips_speed_of_sound',
                 transform_data=None, transform_label=None):
        with open(anno, 'r') as f:
            cohorts = f.readlines()
        self.data_files = []
        self.label_files = []
        self.transform_data = transform_data
        self.transform_label = transform_label
        for cohort in cohorts:
            cohort = cohort.strip()
            self.data_files.extend(sorted(glob(f'{root}/{data_folder}/{cohort}*.npy')))
            self.label_files.extend(sorted(glob(f'{root}/{sos_folder}/{cohort}*.npy')))
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        try: 
            data = np.load(self.data_files[idx])[0]
            label = np.load(self.label_files[idx])[0]
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            data = np.load(self.data_files[0])[0]
            label = np.load(self.label_files[0])[0]
        if self.transform_data:
            data = self.transform_data(data)
        if self.transform_label:
            label = self.transform_label(label)
        return data, label 
        

class ProstateDataset2(Dataset):
    ''' Prostate dataset (bulk file loading)
    This dataset is ~20% slower than ProstateDataset.
    '''

    def __init__(self, anno, file_size=1140, transform_data=None, transform_label=None):
        with open(anno, 'r') as f:
            self.batches = f.readlines()
        self.transform_data = transform_data
        self.transform_label = transform_label
        self.file_size = file_size
    
    def __len__(self):
        return len(self.batches) * self.file_size
    
    def __getitem__(self, idx):
        batch_idx, sample_idx = idx // self.file_size, idx % self.file_size
        batch = self.batches[batch_idx].split('\t')
        try: 
            data = np.load(batch[0], mmap_mode='r')[sample_idx]
            label = np.load(batch[1][:-1], mmap_mode='r')[sample_idx]
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            return np.array([]), np.array([])
        if self.transform_data:
            data = self.transform_data(data)
        if self.transform_label:
            label = self.transform_label(label)
        return data, label 
    

if __name__ == '__main__':
    log_data_min = T.log_transform(-2e-7, k=1).astype('float32')
    log_data_max = T.log_transform(3e-7, k=1).astype('float32')
    transform_data = Compose([
        T.LogTransform(k=1),
        T.MinMaxNormalize(log_data_min, log_data_max)
    ])
    transform_label = Compose([
        T.MinMaxNormalize(1300, 3600)
    ])
    dataset = ProstateDataset(f'../relevant_files/prostate_train.txt', 
                              transform_data=None, 
                              transform_label=None)
    # dataset = ProstateDataset2(f'../relevant_files/prostate_test2.txt', 
    #                         transform_data=transform_data, 
    #                         transform_label=transform_label)
    print(len(dataset))
    print(dataset.data_files[:3], dataset.label_files[:3])
    # data, label = dataset[0]
    # print(data.shape, label.shape)
    # print(data.dtype, label.dtype)

    # dataloader = DataLoader(dataset, batch_size=128, num_workers=12, shuffle=True, pin_memory=True)
    # # dataloader = DataLoader(dataset, batch_size=64, num_workers=12, shuffle=False) # for validation
    # for i, (data, label) in enumerate(dataloader):
    #     print(i, data.shape, label.shape)
    #     print(data.dtype, label.dtype)
    #     break
    # print('Done')