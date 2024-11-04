import os
import pandas as pd
import numpy as np

REPOSITORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPOSITORY, 'data')
CSV_PATH = os.path.join(DATA_DIR, 'metadata.csv')

np.random.seed(42)

def main():
    metadata = pd.read_csv(CSV_PATH, index_col=0)
    
    train_df = pd.DataFrame()  
    valid_df = pd.DataFrame()  
    test_df  = pd.DataFrame()

    for label in ['nonstroke', 'stroke']:
        label_df = metadata[metadata['LABEL']==label]

        pids = label_df['PID'].unique().tolist()
        n_pid = len(pids)
        np.random.shuffle(pids)

        n_train = int(n_pid * 0.8)
        n_valid = int(n_pid * 0.1)

        pid_train = pids[:n_train]
        pid_valid = pids[n_train:n_train+n_valid]
        pid_test = pids[n_train+n_valid:]

        data_train = label_df[label_df['PID'].isin(pid_train)]
        data_valid = label_df[label_df['PID'].isin(pid_valid)]
        data_test = label_df[label_df['PID'].isin(pid_test)]

        train_df = pd.concat([train_df, data_train])
        valid_df = pd.concat([valid_df, data_valid])
        test_df = pd.concat([test_df, data_test])


        # 분할 결과 출력
        print(f"\n라벨 {label} 분포:")
        print(f"화자 수 - Train: {len(pid_train)} ({len(pid_train)/n_pid:.1%})")
        print(f"화자 수 - Valid: {len(pid_valid)} ({len(pid_valid)/n_pid:.1%})")
        print(f"화자 수 - Test: {len(pid_test)} ({len(pid_test)/n_pid:.1%})")
        print(f"파일 수 - Train: {len(data_train)} ({len(data_train)/len(label_df):.1%})")
        print(f"파일 수 - Valid: {len(data_valid)} ({len(data_valid)/len(label_df):.1%})")
        print(f"파일 수 - Test: {len(data_test)} ({len(data_test)/len(label_df):.1%})")
    
    # 전체 데이터 분포 출력
    print("\n전체 데이터 분포:")
    print(f"Train: {len(train_df)} ({len(train_df)/len(metadata):.1%})")
    print(f"Valid: {len(valid_df)} ({len(valid_df)/len(metadata):.1%})")
    print(f"Test: {len(test_df)} ({len(test_df)/len(metadata):.1%})")


    train_df.to_csv(os.path.join(DATA_DIR, 'train.csv'))
    valid_df.to_csv(os.path.join(DATA_DIR, 'valid.csv'))
    test_df.to_csv(os.path.join(DATA_DIR, 'test.csv'))


if __name__ == '__main__':
    main()