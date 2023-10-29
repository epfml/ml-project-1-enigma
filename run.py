import sys
import os
from helpers import load_csv_data


ROOT_DIR = os.path.abspath(os.curdir)
DATA_DIR = ROOT_DIR + "/" + "data"
x_train, x_test, y_train, train_ids, test_ids = load_csv_data(DATA_DIR, sub_sample=True)
