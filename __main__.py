import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, random_split

class AbcDataset(Dataset):
    def __init__(self):
        self.data_list = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"] * 10

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, sample_index):
        target_index = (sample_index + 1) % len(self.data_list)
        return self.data_list[sample_index], self.data_list[target_index]

def main():
    train_dataset, validate_dataset = random_split(AbcDataset(), [0.75, 0.25])
    test_dataset = AbcDataset()

    print("Train (", len(train_dataset), ") :", [train_item for train_item in train_dataset])
    print("Validate (",len(validate_dataset),") :", [validate_item for validate_item in validate_dataset])
    print("Test (",len(test_dataset),") :", [test_item for test_item in test_dataset])

    train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)
    validate_loader = DataLoader(validate_dataset, batch_size = 16, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size = 16, shuffle = False)

    print("Done")

if __name__ == "__main__":
    main()
