import os
import glob
import random
import argparse

import pandas as pd


def split_data(dir,
                ratio=(0.7, 0.15, 0.15)):

    if sum(ratio) != 1:
        raise AttributeError("Sum of the ratio must be equal 1")

    # Get all image paths
    paths = glob.glob(os.path.join(dir, '*'))
    # Filter for each class
    dog_paths = [(path, 0) for path in paths if "dog" in os.path.split(path)[-1]]
    cat_paths = [(path, 1) for path in paths if "cat" in os.path.split(path)[-1]]

    # Split for trainining set
    train_lengths = ( int(ratio[0] * len(dog_paths)), int(ratio[0] * len(cat_paths)) )
    train_paths = dog_paths[:train_lengths[0]] \
                    + cat_paths[:train_lengths[1]]

    # Split for validation set
    valid_lengths = ( int(ratio[1] * len(dog_paths)), int(ratio[1] * len(cat_paths)) )
    valid_paths = dog_paths[train_lengths[0] : (train_lengths[0] + valid_lengths[0])] \
                    + cat_paths[train_lengths[1] : (train_lengths[1] + valid_lengths[1])]

    # Split for test set
    test_paths = dog_paths[(train_lengths[0] + valid_lengths[0]):] \
                    + cat_paths[(train_lengths[1] + valid_lengths[1]):]

    # Shuffle data set
    random.shuffle(train_paths)
    random.shuffle(valid_paths)
    random.shuffle(test_paths)

    return train_paths, valid_paths, test_paths


def save_csv(train_paths,
                valid_paths,
                test_paths):
    # Create dataframe for part
    train_df = pd.DataFrame(data=train_paths, columns=["Path", "Label"])
    valid_df = pd.DataFrame(data=valid_paths, columns=["Path", "Label"])
    test_df = pd.DataFrame(data=test_paths, columns=["Path", "Label"])

    # Save to csv
    train_df.to_csv("data/train.csv", index=False)
    valid_df.to_csv("data/valid.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)



if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--dir', type=str, default="data/train/")
    opt = parse.parse_args()

    train_paths, valid_paths, test_paths = split_data(opt.dir)
    save_csv(train_paths, valid_paths, test_paths)
