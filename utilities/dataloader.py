import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


def check_and_update_csv(old_csv="", new_csv="", img_folders=""):
    # Load CSV file into a DataFrame
    df = pd.read_csv(old_csv,sep=';')
    cams = ["cam1","cam2","cam3","cam4"]

    # Function to check if an image exists in the specified folder
    def image_exists(image_name):
        for cam in cams:
            image_path = os.path.join(img_folders, cam, image_name)
            if os.path.exists(image_path):
                return os.path.exists(image_path)
        return os.path.exists(image_path)

    # Filter out rows where the image does not exist
    df = df[df['imagename'].apply(image_exists)]

    # Save the updated DataFrame back to the CSV file
    df.to_csv(new_csv, index=False,sep=';')


def split_train_val_test(data_dir, data, ratio=[0.7,0.2,0.1], save_as_csv=False):
    cameras = data['camera'].unique()
    train_data, val_data, test_data = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for camera in cameras:
        camera_data = data[data['camera'] == camera]
        
        # Split for each camera (ratio: train:val:test --> 70:20:10)
        train, temp = train_test_split(camera_data, test_size=0.3, random_state=42)
        val, test = train_test_split(temp, test_size=0.33, random_state=42)
        
        train_data = pd.concat([train_data, train], ignore_index=True)
        val_data = pd.concat([val_data, val], ignore_index=True)
        test_data = pd.concat([test_data, test], ignore_index=True)

    if save_as_csv:
        train_data.to_csv(os.path.join(data_dir, 'train_data_t1.csv'), index=False)
        val_data.to_csv(os.path.join(data_dir, 'val_data_t1.csv'), index=False)
        test_data.to_csv(os.path.join(data_dir, 'test_data_t1.csv'), index=False)

    return train_data, val_data, test_data


def split_train_val_test_t2(data_dir, data, save_as_csv=False):
    cameras = data['camera'].unique()
    train_data, val_data, test_data = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for camera in cameras:
        if camera == "cam4":
            test = data[data['camera'] == camera]
            test_data = pd.concat([test_data, test], ignore_index=True)
        else:
            camera_data = data[data['camera'] == camera]
            
            # Split for each camera (ratio: train:val:test --> 70:20:10)
            train, val = train_test_split(camera_data, test_size=0.2, random_state=42)
            # val, test = train_test_split(temp, test_size=0.33, random_state=42)
            
            train_data = pd.concat([train_data, train], ignore_index=True)
            val_data = pd.concat([val_data, val], ignore_index=True)

    if save_as_csv:
        train_data.to_csv(os.path.join(data_dir, 'train_data_t2.csv'), index=False)
        val_data.to_csv(os.path.join(data_dir, 'val_data_t2.csv'), index=False)
        test_data.to_csv(os.path.join(data_dir, 'test_data_t2.csv'), index=False)

    return train_data, val_data, test_data

class ConstructionDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir ,self.dataframe['camera'][idx], self.dataframe['imagename'][idx])
        image = Image.open(img_name).convert('RGB')
        label = self.dataframe['numbered_label'][idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    


def get_dataset(train_df, val_df, test_df, img_dir="", img_size=128):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)), 
        transforms.ToTensor(), # scales to [0.0,1.0]
    ])
    train_dataset = ConstructionDataset(train_df, img_dir, transform=transform)
    val_dataset = ConstructionDataset(val_df, img_dir, transform=transform)
    test_dataset = ConstructionDataset(test_df, img_dir, transform=transform)
    return train_dataset,val_dataset,test_dataset