import os
import re
import glob
import wave
from tqdm import tqdm

import pandas as pd

REPOSITORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(REPOSITORY), 'data')
DATA_DIRS = {
    'stroke':os.path.join(DATA_DIR, 'stroke'),
    'nonstroke':os.path.join(DATA_DIR, 'nonstroke_VOTE400_Read', 'Audio')
}
SAVE_PATH = os.path.join(REPOSITORY, 'data', 'metadata.csv')

print(f'REPOSITORY: {REPOSITORY}')
print(f'DATA_DIR: {DATA_DIR}')
print(f'DATA_DIRS: {DATA_DIRS}')

def get_duration(path):
    with wave.open(path, 'rb') as wav_file:
        # 프레임 수와 프레임레이트를 통해 길이 계산
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
        return duration
    
def main():
    if os.path.exists(SAVE_PATH):
        print(f'{SAVE_PATH} already exists.') 
    else:
        metadata = []
        for path in tqdm(glob.glob(pathname='*/*.wav', root_dir = DATA_DIRS['nonstroke'], recursive=True)):
            duration  = get_duration(os.path.join(DATA_DIRS['nonstroke'], path))
            if os.sep in path:
                loc, fname = path.split(os.sep)
                loc = re.match(r'[A-Z]+', loc).group()
            pid = fname.split('_')[1]
            metadata.append([pid, loc, fname, duration, 'nonstroke'])

        for path in tqdm(glob.glob(pathname = '*/*.wav', root_dir= DATA_DIRS['stroke'], recursive=True)):
            duration  = get_duration(os.path.join(DATA_DIRS['stroke'], path))
            if os.sep in path:
                fname = path.split(os.sep)[-1]
            info = fname.split('_')[0].split('-')[4:]
            if len(info) == 6:
                name, _, _, sex, age, loc = info
            elif len(info) == 5:
                name, _, sex, age, loc = info
            # print(fname.split('_')[0].split('-')[4:])
            pid = '_'.join([name, sex, age])
            metadata.append([pid, loc, fname, duration, 'stroke'])

        metadata = pd.DataFrame(metadata, columns=['PID', 'LOC', 'FNAME', 'DUR(s)', 'LABEL'])
        metadata['LOC'] = metadata['LOC'].replace({'SU':'SE', 'kk': 'KK', 'KK3':'KK'})
        metadata.to_csv(SAVE_PATH)

if __name__ == '__main__':
    main()
