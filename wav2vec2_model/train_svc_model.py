import os
import json
from tqdm import tqdm

import librosa
import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.metrics import classification_report

import torch
from datasets import Dataset
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

import joblib


root_path = os.path.abspath('.')
dirname = 'wav2vec2_model'
dataset_dir = os.path.join(root_path, dirname, 'dataset')
output_dir = os.path.join(root_path, dirname, 'output')
wav2vec2_dir = os.path.join(root_path, dirname, 'k_wav2vec2')
batch_size=4

def get_datasets(dataset_dir):
    train_path = os.path.join(dataset_dir, 'train.csv')
    valid_path = os.path.join(dataset_dir,'valid.csv')
    # test_path = os.path.join(dataset_dir,'test.csv')

    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)

    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    # test_dataset = Dataset.from_pandas(test_df)
    return train_dataset, valid_dataset


class Preprocessor:
     def __init__(self, input_column, output_column, label_list, max_len=None):
          self.input_column = input_column
          self.output_column = output_column
          self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base-960h')
          self.target_sr = self.feature_extractor.sampling_rate
          self.label_list = label_list
          self.max_len = max_len

     def __call__(self, examples):
          speech_list = [self.audio_to_array(path, self.target_sr) for path in examples[self.input_column]]
          target_list = [self.label_to_id(label, self.label_list) for label in examples[self.output_column]]

          result = self.feature_extractor(speech_list, sampling_rate=self.target_sr)

          result['input_values'] = self.pad_array(result['input_values'])
          result['labels'] = np.array(list(target_list))
          return result

     def audio_to_array(self, path, sr):
          y, _ = librosa.load(path, sr=sr)
          return y

     def label_to_id(self, label, label_list):
          if len(label_list) > 0:
               return label_list.index(label) if label in label_list else -1
          return label
     
     def get_max_len(self, speech_list):
          speech_lengths = np.array([len(speech_array) for speech_array in speech_list])
          self.max_len = int(np.quantile(speech_lengths, q=0.75))

     def pad_array(self, input_values):
        if self.max_len is None:
             self.get_max_len(input_values)
             
        padded_input_values = []
        for x in input_values:
            padding_size =  self.max_len - len(x) if len(x) < self.max_len else 0
            if padding_size  > 0 :
                padded_input_values.append(np.pad(x, (0, padding_size), 'constant'))
            else:
                padded_input_values.append(x[:self.max_len])
        return np.array(padded_input_values)


def extract_hidden_states(input_values, model, index=0):
    input_values = torch.tensor(input_values, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(input_values, output_hidden_states=True)
    hidden_states = outputs.hidden_states[index]
    pooled_hidden_vectors = hidden_states.mean(dim=1).cpu().numpy()
    return pooled_hidden_vectors

def extract_features_and_labels(dataset, wav2vec2_model, batch_size=4):
    hidden_vector_list = []
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch= dataset[i:i + batch_size]

        hidden_vectors = extract_hidden_states(batch['input_values'], wav2vec2_model)
        hidden_vector_list.append(hidden_vectors)
        
    X = np.concatenate(hidden_vector_list, axis=0)
    y = np.array(dataset['labels'])
    return X, y

def main():
    train_dataset, valid_dataset = get_datasets(dataset_dir=dataset_dir)
    label_list = train_dataset.unique('label')

    preprocessor = Preprocessor('path', 'label', label_list)
    
    train_dataset = train_dataset.map(
        preprocessor,
        batch_size = batch_size,
        batched=True,
        remove_columns=['path', 'name', 'Unnamed: 0']
    )

    valid_dataset = valid_dataset.map(
        preprocessor,
        batch_size = batch_size,
        batched=True,
        remove_columns=['path', 'name', 'Unnamed: 0']
    )
    
    wav2vec2_model = Wav2Vec2Model.from_pretrained(wav2vec2_dir)

    X_train, y_train = extract_features_and_labels(train_dataset, wav2vec2_model)
    X_valid, y_valid = extract_features_and_labels(valid_dataset, wav2vec2_model)

    svc_model = svm.SVC()
    svc_model.fit(X_train, y_train)

    y_pred = svc_model.predict(X_valid)

    report = classification_report(y_valid, y_pred, output_dict=True)
  
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    model_path = os.path.join(output_dir, 'w2v2_svc_model.joblib')
    joblib.dump(svc_model, model_path)

    model_report_path = os.path.join(output_dir, 'model_report.json')
    stroke_index = str(label_list.index('stroke'))
    with open(model_report_path, 'w') as f:
        json.dump(report[stroke_index], f)

if __name__ == '__main__':
    main()