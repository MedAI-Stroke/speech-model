import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

USE_SUBSET = True
FILTER_CSV_PATH = 'preprocessed_data.csv'


def make_dataset_df(dataset_dir='dataset', 
                    class_names:list=['stroke', 'non-stroke']):
    df = pd.DataFrame()

    for class_name in class_names:
        class_dirpath = os.path.join(os.path.abspath('.'), dataset_dir, class_name)
        paths= pd.Series(glob.glob(class_dirpath + "/**/*.wav"))
        names = paths.str.split(os.sep).str[-1].str.split('.').str[0]
        names = names.str.replace('ID-01-11-N-', '').str.replace(r'[a-zA-Z]{2}_chunk', '', regex=True)
        labels = class_name

        class_df = pd.DataFrame({
            'path' : paths,
            'name':names,
            'label':labels
        })
        df = pd.concat([df, class_df])
    return df

def filter_data(dataset, filter_csv_path):
    filter_df= pd.read_csv(filter_csv_path)['fname']
    dataset['fname'] = dataset['path'].str.split(os.sep).str[-1]
    filtered_dataset = dataset[dataset['fname'].isin(filter_df)]
    return filtered_dataset


def main():
    root_path = os.path.abspath('.')
    dataset_dir = os.path.join(root_path, 'dataset')

    df = make_dataset_df(dataset_dir=dataset_dir)

    if USE_SUBSET == True:
        df = filter_data(df, FILTER_CSV_PATH)

    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    valid_df, test_df = train_test_split(valid_df, test_size=0.5, random_state=42, stratify=valid_df['label'])

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        
    train_df.to_csv(os.path.join(dataset_dir, 'train.csv'))
    valid_df.to_csv(os.path.join(dataset_dir,'valid.csv'))
    test_df.to_csv(os.path.join(dataset_dir,'test.csv'))

if __name__ == '__main__':
    main()
