import pandas as pd

from src.data_loader.data_loader import DataLoader
from src.data_loader.data_preprocess import DataPreprocess


def train():
    dl = DataLoader()
    data_train = dl.train_test_load()
    dp = DataPreprocess()
    dp.load_data(data_train)
    data_preprocessed = dp.choose_variables()
    print(data_preprocessed)


if __name__ == "__main__":
    train()
