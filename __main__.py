import numpy as np

import torch
import torch.nn as nn

from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split

class AbcDataset(Dataset):
    def __init__(self):
        sample = np.arange(0, 1, 1/26, dtype = np.float32)
        self.data_list = np.concat((sample, sample, sample, sample, sample, sample, sample, sample, sample, sample))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, sample_index):
        target_index = (sample_index + 1) % len(self.data_list)
        return self.data_list[sample_index], self.data_list[target_index]

class AbcModule(nn.Module):
    def __init__(self):
        self.INPUT_LENGTH = 1
        self.OUTPUT_LENGTH = 1

        super().__init__()

        self.layer_1 = nn.Linear(self.INPUT_LENGTH, 26 * 10)
        self.activation_1 = nn.ReLU()
        self.layer_2 = nn.Linear(26 * 10, 26 * 5)
        self.activation_2 = nn.ReLU()
        self.layer_3 = nn.Linear(26 * 5, 26)
        self.activation_3 = nn.ReLU()
        self.layer_4 = nn.Linear(26, self.OUTPUT_LENGTH)

        self.model = nn.Sequential(
            self.layer_1,
            self.activation_1,
            self.layer_2,
            self.activation_2,
            self.layer_3,
            self.activation_3,
            self.layer_4)

    def forward(self, x):
        y = self.model(x)
        return y

def main():
    BATCH_SIZE = 16

    train_dataset, validate_dataset = random_split(AbcDataset(), [0.75, 0.25])
    test_dataset = AbcDataset()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AbcModule().to(device)

    loss_model = nn.HuberLoss()
    optim_model = Adam(model.parameters(), lr = 0.001)

    input = torch.rand([BATCH_SIZE, model.INPUT_LENGTH], dtype = torch.float32).to(device)
    target = torch.rand([BATCH_SIZE, model.INPUT_LENGTH], dtype = torch.float32).to(device)

    output = model(input)
    loss = loss_model(output, target)

    print(input.shape, input)
    print(target.shape, target)
    print(output.shape, output)
    print(loss)

    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
    validate_loader = DataLoader(validate_dataset, batch_size = BATCH_SIZE, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False)

    EPOCHS = 1000

    for epoch in range(EPOCHS):
        model.train()
        for x, targets in train_loader:
            x = x.reshape(-1, 1).to(device)
            targets = targets.reshape(-1, 1).to(device)

            y = model(x)
            loss = loss_model(y, targets)

            optim_model.zero_grad()
            loss.backward()
            optim_model.step()

        model.eval()
        with torch.no_grad():
            for x, targets in validate_loader:
                x = x.reshape(-1, 1).to(device)
                targets = targets.reshape(-1, 1).to(device)

                y = model(x)
                loss = loss_model(y, targets)

    sample = torch.tensor([0, 1, 2, 3, 22, 23, 24, 25], dtype = torch.float32)

    input = (sample / 26).reshape(-1, 1).to(device)
    output = model(input)

    input = torch.round(input * 26)
    output = torch.round(output * 26)

    print(input, output)
    print("Done")

if __name__ == "__main__":
    main()
