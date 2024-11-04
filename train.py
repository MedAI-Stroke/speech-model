import os
import tensorflow as tf
import numpy as np
import pandas as pd
import h5py
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

class AudioDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, h5_path, dataframe, batch_size=32, shuffle=True):
        self.h5_path = h5_path
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.shuffle = shuffle
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
        
        if shuffle:
            np.random.shuffle(self.indexes)

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

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
    def __del__(self):
        self.h5_file.close()

# AUROC 계산을 위한 커스텀 콜백
class AUROCCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, save_path):
        super().__init__()
        self.validation_data = validation_data
        self.save_path = save_path
        self.best_auroc = 0
        
    def on_epoch_end(self, epoch, logs={}):
        # 전체 검증 세트에 대한 예측
        y_pred = []
        y_true = []
        
        for i in range(len(self.validation_data)):
            x, y = self.validation_data[i]
            pred = self.model.predict(x, verbose=0)
            y_pred.extend(pred.flatten())
            y_true.extend(y)
        
        # AUROC 계산
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auroc = auc(fpr, tpr)
        
        logs['val_auroc'] = auroc
        
        # 최고 AUROC 모델 저장
        if auroc > self.best_auroc:
            self.best_auroc = auroc
            self.model.save(self.save_path)
            
        print(f' - val_auroc: {auroc:.4f}')

def create_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                     input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model


def create_dual_perspective_model(input_shape):
    input_layer = layers.Input(shape=input_shape)
    
    # 시간축 중심 경로 (temporal path)
    temporal = layers.Conv2D(32, (7, 3), padding='same', activation='relu')(input_layer)
    temporal = layers.BatchNormalization()(temporal)
    temporal = layers.MaxPooling2D((2, 2))(temporal)
    
    temporal = layers.Conv2D(64, (7, 3), padding='same', activation='relu')(temporal)
    temporal = layers.BatchNormalization()(temporal)
    temporal = layers.MaxPooling2D((2, 2))(temporal)
    
    # 주파수축 중심 경로 (spectral path)
    spectral = layers.Conv2D(32, (3, 7), padding='same', activation='relu')(input_layer)
    spectral = layers.BatchNormalization()(spectral)
    spectral = layers.MaxPooling2D((2, 2))(spectral)
    
    spectral = layers.Conv2D(64, (3, 7), padding='same', activation='relu')(spectral)
    spectral = layers.BatchNormalization()(spectral)
    spectral = layers.MaxPooling2D((2, 2))(spectral)
    
    # 특징 결합 (feature fusion)
    merged = layers.Concatenate()([temporal, spectral])
    
    # 공통 특징 추출 (shared feature extraction)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(merged)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # 전역 특징 추출 및 분류
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    output = layers.Dense(1, activation='sigmoid')(x)
    
    return tf.keras.Model(inputs=input_layer, outputs=output)

def create_tri_perspective_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    
     # Low-order MFCC stream (1-4)
    low_mfcc = layers.Lambda(lambda x: x[:, :4, :, :])(inputs)
    low_stream = layers.Conv2D(8, (2, 3), activation='relu', padding='same')(low_mfcc)
    low_stream = layers.MaxPooling2D((2, 2))(low_stream)
    low_stream = layers.Conv2D(16, (2, 3), activation='relu', padding='same')(low_stream)
    low_stream = layers.MaxPooling2D((2, 2))(low_stream)
    low_stream = layers.BatchNormalization()(low_stream)
    low_stream = layers.GlobalAveragePooling2D()(low_stream)
    
    # High-order MFCC stream (5-13)
    high_mfcc = layers.Lambda(lambda x: x[:, 4:, :, :])(inputs)
    high_stream = layers.Conv2D(12, (3, 3), activation='relu', padding='same')(high_mfcc)
    high_stream = layers.MaxPooling2D((2, 2))(high_stream)
    high_stream = layers.Conv2D(24, (3, 3), activation='relu', padding='same')(high_stream)
    high_stream = layers.MaxPooling2D((2, 2))(high_stream)
    high_stream = layers.BatchNormalization()(high_stream)
    high_stream = layers.GlobalAveragePooling2D()(high_stream)
    
    # Full MFCC stream (1-13)
    full_stream = layers.Conv2D(8, (4, 5), activation='relu', padding='same')(inputs)
    full_stream = layers.MaxPooling2D((2, 2))(full_stream)
    full_stream = layers.Conv2D(16, (4, 5), activation='relu', padding='same')(full_stream)
    full_stream = layers.MaxPooling2D((2, 2))(full_stream)
    full_stream = layers.BatchNormalization()(full_stream)
    full_stream = layers.GlobalAveragePooling2D()(full_stream)
    
    # Combine streams
    merged = layers.Concatenate()([low_stream, high_stream, full_stream])
    
    # Final dense layers (reduced size)
    x = layers.Dropout(0.3)(merged)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def plot_training_history(history, save_dir):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
    
    # 손실 그래프
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # 정확도 그래프
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    # AUROC 그래프
    ax3.plot(history.history['val_auroc'], label='Validation AUROC')
    ax3.set_title('Model AUROC')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('AUROC')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()

def save_model_architecture(model, save_dir):
    """
    모델 아키텍처를 텍스트 파일로 저장
    
    Args:
        model: Keras 모델
        save_dir: 저장할 디렉토리 경로
    """
    # 모델 구조를 문자열로 캡처
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    
    # architecture_info.txt 파일에 저장
    with open(os.path.join(save_dir, 'architecture_info.txt'), 'w') as f:
        # 모델 타입 저장
        f.write(f"Model Type: {model.__class__.__name__}\n")
        if hasattr(model, 'name'):
            f.write(f"Model Name: {model.name}\n")
        f.write("\nModel Architecture:\n")
        f.write("=" * 50 + "\n")
        f.write('\n'.join(model_summary))
        
        # 모델 파라미터 수 계산
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        f.write("\nModel Parameters:\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total params: {total_params:,}\n")
        f.write(f"Trainable params: {trainable_params:,}\n")
        f.write(f"Non-trainable params: {non_trainable_params:,}\n")

def calculate_auroc(model, generator):
    y_pred = []
    y_true = []
    
    for i in range(len(generator)):
        x, y = generator[i]
        pred = model.predict(x, verbose=0)
        y_pred.extend(pred.flatten())
        y_true.extend(y)
    
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return auc(fpr, tpr)


def save_performance_metrics(model, train_generator, valid_generator, test_generator, save_dir):
    """
    모든 데이터셋에 대한 성능 지표 계산 및 저장
    
    Args:
        model: Keras 모델
        train_generator: 학습 데이터 제너레이터
        valid_generator: 검증 데이터 제너레이터
        test_generator: 테스트 데이터 제너레이터
        save_dir: 저장할 디렉토리 경로
    """
    # 각 데이터셋에 대한 AUROC 계산
    train_auroc = calculate_auroc(model, train_generator)
    valid_auroc = calculate_auroc(model, valid_generator)
    test_auroc = calculate_auroc(model, test_generator)
    
    # 각 데이터셋에 대한 손실값과 정확도 계산
    train_loss, train_acc = model.evaluate(train_generator, verbose=0)
    valid_loss, valid_acc = model.evaluate(valid_generator, verbose=0)
    test_loss, test_acc = model.evaluate(test_generator, verbose=0)
    
    # metrics.txt 파일에 결과 저장
    with open(os.path.join(save_dir, 'best_model_metrics.txt'), 'w') as f:
        f.write("Best Model Performance Metrics\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Training Set Metrics:\n")
        f.write("-" * 20 + "\n")
        f.write(f"AUROC: {train_auroc:.4f}\n")
        f.write(f"Accuracy: {train_acc:.4f}\n")
        f.write(f"Loss: {train_loss:.4f}\n\n")
        
        f.write("Validation Set Metrics:\n")
        f.write("-" * 20 + "\n")
        f.write(f"AUROC: {valid_auroc:.4f}\n")
        f.write(f"Accuracy: {valid_acc:.4f}\n")
        f.write(f"Loss: {valid_loss:.4f}\n\n")
        
        f.write("Test Set Metrics:\n")
        f.write("-" * 20 + "\n")
        f.write(f"AUROC: {test_auroc:.4f}\n")
        f.write(f"Accuracy: {test_acc:.4f}\n")
        f.write(f"Loss: {test_loss:.4f}\n")

def evaluate_model(model, generator, save_dir):
    # 예측값 생성
    y_pred = []
    y_true = []
    
    for i in range(len(generator)):
        x, y = generator[i]
        pred = model.predict(x, verbose=0)
        y_pred.extend(pred.flatten())
        y_true.extend(y)
    
    # AUROC 계산 및 ROC 곡선 그리기
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
    plt.close()
    
    return roc_auc

def main(model_type):
    # 설정
    REPOSITORY = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(REPOSITORY, 'data')
    H5_PATH = os.path.join(DATA_DIR, 'mfcc_features.h5')
    BATCH_SIZE = 32
    EPOCHS = 50
    
    # 결과 저장을 위한 디렉토리 생성
    SAVE_DIR = os.path.join(REPOSITORY, 'results')
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 데이터 로드
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    valid_df = pd.read_csv(os.path.join(DATA_DIR, 'valid.csv'))
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    
    # 데이터 제너레이터 생성
    train_generator = AudioDataGenerator(H5_PATH, train_df, batch_size=BATCH_SIZE)
    valid_generator = AudioDataGenerator(H5_PATH, valid_df, batch_size=BATCH_SIZE, shuffle=False)
    test_generator = AudioDataGenerator(H5_PATH, test_df, batch_size=BATCH_SIZE, shuffle=False)
    
    # 입력 형태 계산
    sample_mfcc = train_generator[0][0][0]
    input_shape = sample_mfcc.shape
    print(f"Input shape: {input_shape}")
    
    # 모델 생성
    if model_type == 'baseline':
        model = create_model(input_shape)
    elif model_type == 'dual':
        model = create_dual_perspective_model(input_shape)
    elif model_type == 'tri':
        model = create_tri_perspective_model(input_shape)
    model.summary()
    save_model_architecture(model, SAVE_DIR)
    
    # 모델 컴파일
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # 콜백 설정
    callbacks = [
        AUROCCallback(valid_generator, os.path.join(SAVE_DIR, 'best_model.h5')),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(SAVE_DIR, 'training_log.csv')
        )
    ]
    
    # 클래스 가중치 계산
    class_counts = train_df['LABEL'].value_counts()
    total = len(train_df)
    class_weight = {
        0: total / (2 * class_counts['nonstroke']),
        1: total / (2 * class_counts['stroke'])
    }
    print("Class weights:", class_weight)
    
    # 모델 학습
    history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight
    )
    
    # 학습 결과 시각화
    plot_training_history(history, SAVE_DIR)
    
     
    # best model의 성능 지표 저장
    save_performance_metrics(
        model,
        train_generator,
        valid_generator,
        test_generator,
        SAVE_DIR
    )

if __name__ == '__main__':
    main(model_type='baseline')