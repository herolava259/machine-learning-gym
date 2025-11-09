import pandas as pd

def load_new_group_dataset():
    return pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json")