import warnings
warnings.filterwarnings(action='ignore')

import matplotlib.pyplot as plt
import pandas as pd

# use 50 top tags
TAGS = ['guitar','classical', 'slow','techno','strings','drums','electronic','rock',
        'fast','piano','ambient','beat','violin','vocal','synth','female','indian',
        'opera','male','singing','vocals','no vocals','harpsichord','loud','quiet',
        'flute', 'woman', 'male vocal', 'no vocal', 'pop','soft','sitar', 'solo',
        'man', 'classic', 'choir', 'voice', 'new age', 'dance', 'male voice',
        'female vocal', 'beats', 'harp', 'cello', 'no voice', 'weird', 'country',
        'metal', 'female voice', 'choral']


def check_data():
    df = pd.read_csv("./data/annotations_final.csv", sep="\t", index_col=0)
    item = df.iloc[0]
    # check annotated tags
    item[item != 0] 
    # check data distribution
    df[TAGS].sum().plot.bar(figsize=(18,6),rot=60) 
    plt.show()

    return df


def preprocess_data(df):
    train = []
    valid = []
    test = []
    id_to_path = {}
    for idx in range(len(df)):
        item = df.iloc[idx]
        id = item.name
        path = item['mp3_path']
        folder = path.split("/")[0]
        id_to_path[id] = path
        if folder in "012ab":
            train.append(id)# split = "train"
        elif folder == "c":    
            valid.append(id)# split = "valid"
        elif folder in "d":
            test.append(id)# split = "test"

    total = len(train) + len(valid) + len(test)
    print(f'Total number of data: {total}')
    print(f'Percentage of training data: {int(len(train) / total * 100)}%')
    print(f'Percentage of validation data: {int(len(valid) / total * 100)}%')
    print(f'Percentage of test data: {int(len(test) / total * 100)}%')

    df = df[TAGS]
    df_train = df.loc[train]
    df_valid = df.loc[valid]
    df_test = df.loc[test]
    
    return df_train, df_valid, df_test, id_to_path

