import os
import glob
from tqdm import tqdm
import librosa
import numpy as np
import pandas as pd
import h5py
from multiprocessing import Pool
from functools import partial
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import multiprocessing

REPOSITORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(os.path.dirname(REPOSITORY), 'data')
AUDIO_DIRS = {
    'stroke': os.path.join(AUDIO_DIR, 'stroke'),
    'nonstroke': os.path.join(AUDIO_DIR, 'nonstroke_VOTE400_Read')
}
DATA_DIR = os.path.join(REPOSITORY, 'data')
PNG_SAVE_DIR = os.path.join(DATA_DIR, 'visualization')
CSV_PATH = os.path.join(DATA_DIR, 'metadata.csv')
H5_PATH = os.path.join(DATA_DIR, 'mfcc_features.h5')

def rms_normalize(audio, target_dB=-20):
    rms = np.sqrt(np.mean(audio**2))
    target_rms = 10**(target_dB/20)
    return audio * (target_rms/rms)

def adaptive_preemphasis(audio, sr):
    spec = np.abs(librosa.stft(audio))
    freq_ratio = np.mean(spec[spec.shape[0]//2:]) / np.mean(spec[:spec.shape[0]//2])
    
    if freq_ratio < 0.1:
        alpha = 0.97
    elif freq_ratio < 0.3:
        alpha = 0.95
    else:
        alpha = 0.90
        
    return np.append(audio[0], audio[1:] - alpha * audio[:-1]), alpha

def extract_mfcc(audio_info):
    """오디오 파일에서 MFCC 특성 추출"""
    path, fname = audio_info
    
    try:
        # 1. Load audio
        audio, sr = librosa.load(path, sr=None)

        # 2. Standardization of Sampling Rate
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        # 3. Volume Normalization
        audio = rms_normalize(audio)

        # 4. Pre-emphasis
        audio, _ = adaptive_preemphasis(audio, sr=16000)

        # 5. 묵음 제거
        intervals = librosa.effects.split(audio, top_db=20)
        audio = np.concatenate([audio[start:end] for start, end in intervals])

        # 6. 오디오 길이 표준화
        target_length = 320000 # 20초 * 16000 sr
        if len(audio) < target_length : 
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            audio = audio[:target_length]

        # 6. MFCC 추출
        mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13, hop_length= 512)

        # 7. MFCC 정규화
        mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / \
               (np.std(mfcc, axis=1, keepdims=True) + 1e-8)

        return fname, mfcc, True
    except Exception as e:
        print(f"Error processing {fname}: {str(e)}")
        return fname, None, False
    
class MFCCDataManager:
    def __init__(self, base_path):
        self.base_path = base_path
        self.processed_count = 0
        self.success_count = 0  # 성공적으로 처리된 파일 수 추가
        self.total_files = 0
        self.start_time = None
        self.failed_files = []
    
    def calculate_estimated_time(self):
        """예상 완료 시간 계산"""
        if self.processed_count == 0:
            return "Calculating..."
        
        elapsed_time = time.time() - self.start_time
        files_per_second = self.processed_count / elapsed_time
        remaining_files = self.total_files - self.processed_count
        estimated_seconds = remaining_files / files_per_second
        
        completion_time = datetime.now() + timedelta(seconds=estimated_seconds)
        return completion_time.strftime("%Y-%m-%d %H:%M:%S")

    def save_batch(self, results):
        """배치 단위로 MFCC 특성을 HDF5 파일에 저장"""
        with h5py.File(os.path.join(self.base_path, 'mfcc_features.h5'), 'a') as hf:
            for result in results:
                if isinstance(result, tuple) and len(result) == 3:
                    fname, mfcc_data, success = result
                    if success and mfcc_data is not None:
                        try:
                            key = os.path.splitext(fname)[0]  # 확장자 제거
                            if key in hf:
                                del hf[key]
                            # mfcc_data가 numpy array인지 확인하고 저장
                            mfcc_array = np.asarray(mfcc_data)
                            hf.create_dataset(key, data=mfcc_array, compression='gzip')
                            self.success_count += 1  # 성공적으로 저장된 경우 카운트
                        except Exception as e:
                            print(f"Error saving {fname}: {str(e)}")
                            self.failed_files.append(fname)
                    else:
                        self.failed_files.append(fname)
                else:
                    print(f"Invalid result format: {result}")
    
    def process_files(self, file_list, num_processes=4, batch_size=50):
        """멀티프로세싱을 사용하여 파일 처리"""
        self.total_files = len(file_list)
        self.start_time = time.time()
        self.processed_count = 0
        self.success_count = 0
        self.failed_files = []
        
        print(f"\nTotal files to process: {self.total_files}")
        print(f"Using {num_processes} processes")
        
        # 이미 처리된 파일 확인
        processed_files = set()
        h5_path = os.path.join(self.base_path, 'mfcc_features.h5')
        if os.path.exists(h5_path):
            with h5py.File(h5_path, 'r') as hf:
                processed_files = set(hf.keys())
        
        # 처리가 필요한 파일만 필터링
        remaining_files = [
            (path, fname) for path, fname in file_list 
            if os.path.splitext(fname)[0] not in processed_files
        ]
        
        if len(remaining_files) < len(file_list):
            print(f"\nFound {len(file_list) - len(remaining_files)} already processed files")
            print(f"Remaining files to process: {len(remaining_files)}")
        
        if not remaining_files:
            print("No files to process.")
            return
        
        with Pool(processes=num_processes) as pool:
            pbar = tqdm(total=len(remaining_files), desc="Processing files")
            
            for i in range(0, len(remaining_files), batch_size):
                batch = remaining_files[i:i + batch_size]
                results = pool.map(extract_mfcc, batch)
                
                # 결과 저장
                self.save_batch(results)
                
                # 진행상황 업데이트
                self.processed_count += len(batch)
                pbar.update(len(batch))
                
                # 예상 완료 시간 표시
                est_completion = self.calculate_estimated_time()
                pbar.set_postfix({
                    'Processed': f"{self.processed_count}/{self.total_files}",
                    'Success': f"{self.success_count}/{self.processed_count}",
                    'Est. Completion': est_completion
                })
            
            pbar.close()
        
        # 처리 완료 통계
        end_time = time.time()
        total_time = end_time - self.start_time
        
        print("\nProcessing Summary:")
        print(f"Total time: {timedelta(seconds=int(total_time))}")
        if self.processed_count > 0:
            print(f"Average time per file: {total_time/self.processed_count:.2f} seconds")
        print(f"Files processed: {self.processed_count}")
        print(f"Successfully processed: {self.success_count}")
        
        if self.failed_files:
            print(f"\nFailed files ({len(self.failed_files)}):")
            for fname in self.failed_files:
                print(f"- {fname}")


def main():    
    # 파일 목록 생성
    file_list = []
    for label in ['stroke', 'nonstroke']:
        root_dir = AUDIO_DIRS[label]
        paths = glob.glob(f'**/*.wav', root_dir=root_dir, recursive=True)
        file_list += [(os.path.join(root_dir, path), path.split(os.sep)[-1]) for path in paths]

    # CPU 코어 수의 75%만 사용
    num_cores = multiprocessing.cpu_count()
    num_processes = max(1, int(num_cores * 0.75))  # 24개 코어의 경우 18개 사용
    
    print(f"\n사용 가능한 전체 코어 수: {num_cores}")
    print(f"사용할 프로세스 수: {num_processes}")
    
    # 예상 처리 시간 계산
    sample_size = min(5, len(file_list))
    print(f"\nTesting with {sample_size} files to estimate processing time...")
    
    manager = MFCCDataManager(DATA_DIR)
    # 테스트는 적은 수의 프로세스로
    manager.process_files(file_list[:sample_size], num_processes=4, batch_size=5)
    
    avg_time = (time.time() - manager.start_time) / sample_size
    total_estimated_time = timedelta(seconds=int(avg_time * len(file_list) / (num_processes/4)))  # 프로세스 수 증가 고려
    print(f"\nEstimated total processing time: {total_estimated_time}")
    
    response = input("\nContinue with full processing? (y/n): ")
    if response.lower() == 'y':
        manager = MFCCDataManager(DATA_DIR)
        # 실제 처리는 더 많은 프로세스로
        manager.process_files(file_list, num_processes=num_processes, batch_size=50)
    else:
        print("Processing cancelled")

if __name__ == '__main__':
    main()