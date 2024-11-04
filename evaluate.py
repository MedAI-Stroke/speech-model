import os
import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import precision_score, recall_score, accuracy_score

class AudioDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, h5_path, dataframe, batch_size=32):
        self.h5_path = h5_path
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.indexes = np.arange(len(dataframe))
        
        # HDF5 파일 열기
        self.h5_file = h5py.File(h5_path, 'r')
        
        # 패딩을 위한 최대 길이 계산
        max_length = 0
        for fname in dataframe['FNAME']:
            key = fname.split('.')[0]
            if key in self.h5_file:
                length = self.h5_file[key].shape[1]
                max_length = max(max_length, length)
        self.max_length = max_length

    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_df = self.dataframe.iloc[batch_indexes]
        
        batch_x = []
        batch_y = []
        
        for _, row in batch_df.iterrows():
            fname = row['FNAME']
            key = fname.split('.')[0]
            
            if key in self.h5_file:
                mfcc = self.h5_file[key][()]
                
                if mfcc.shape[1] < self.max_length:
                    pad_width = ((0, 0), (0, self.max_length - mfcc.shape[1]))
                    mfcc = np.pad(mfcc, pad_width, mode='constant')
                
                mfcc = np.expand_dims(mfcc, axis=-1)
                
                batch_x.append(mfcc)
                batch_y.append(1 if row['LABEL'] == 'stroke' else 0)
        
        return np.array(batch_x), np.array(batch_y)
            
    def __del__(self):
        self.h5_file.close()

def get_predictions(model, generator):
    y_pred = []
    y_true = []
    
    for i in range(len(generator)):
        x, y = generator[i]
        pred = model.predict(x, verbose=0)
        y_pred.extend(pred.flatten())
        y_true.extend(y)
    
    return np.array(y_true), np.array(y_pred)

def calculate_metrics(y_true, y_pred, threshold=0.5):
    y_pred_binary = (y_pred >= threshold).astype(int)
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    accuracy = accuracy_score(y_true, y_pred_binary)
    return precision, recall, accuracy

def plot_roc_curves(results, save_path):
    plt.figure(figsize=(12, 8))
    
    colors = {'train': 'blue', 'valid': 'darkorange', 'test': 'green'}
    
    for dataset_name, metrics in results.items():
        fpr, tpr, _ = roc_curve(metrics['true'], metrics['pred'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[dataset_name], lw=2,
                label=f'{dataset_name.capitalize()} ROC (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_pr_curves(results, save_path):
    plt.figure(figsize=(12, 8))
    
    colors = {'train': 'blue', 'valid': 'darkorange', 'test': 'green'}
    
    for dataset_name, metrics in results.items():
        precision, recall, _ = precision_recall_curve(metrics['true'], metrics['pred'])
        avg_precision = average_precision_score(metrics['true'], metrics['pred'])
        plt.plot(recall, precision, color=colors[dataset_name], lw=2,
                label=f'{dataset_name.capitalize()} PR (AP = {avg_precision:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.grid(True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 경로 설정
    REPOSITORY = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(REPOSITORY, 'data')
    RESULTS_DIR = os.path.join(REPOSITORY, 'results')
    H5_PATH = os.path.join(DATA_DIR, 'mfcc_features.h5')
    MODEL_PATH = os.path.join(RESULTS_DIR, 'best_model.keras')
    
    # 결과 저장 디렉토리 생성
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 데이터 로드
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    valid_df = pd.read_csv(os.path.join(DATA_DIR, 'valid.csv'))
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    
    # 데이터 제너레이터 생성
    train_generator = AudioDataGenerator(H5_PATH, train_df, batch_size=32)
    valid_generator = AudioDataGenerator(H5_PATH, valid_df, batch_size=32)
    test_generator = AudioDataGenerator(H5_PATH, test_df, batch_size=32)
    
    # 모델 로드
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # 각 데이터셋에 대한 예측 수행
    results = {}
    for name, generator in [('train', train_generator), 
                          ('valid', valid_generator), 
                          ('test', test_generator)]:
        y_true, y_pred = get_predictions(model, generator)
        precision, recall, accuracy = calculate_metrics(y_true, y_pred)
        
        results[name] = {
            'true': y_true,
            'pred': y_pred,
            'precision': precision,
            'recall': recall,
             'accuracy': accuracy
        }
    
    # 메트릭 출력
    print("\nEvaluation Metrics:")
    print("-" * 50)
    for dataset_name, metrics in results.items():
        print(f"\n{dataset_name.capitalize()} Set:")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"AUC: {auc(roc_curve(metrics['true'], metrics['pred'])[0], roc_curve(metrics['true'], metrics['pred'])[1]):.4f}")
    
    # ROC 커브와 PR 커브 그리기
    plot_roc_curves(results, os.path.join(RESULTS_DIR, 'roc_curves_comparison.png'))
    plot_pr_curves(results, os.path.join(RESULTS_DIR, 'pr_curves_comparison.png'))

if __name__ == '__main__':
    main()