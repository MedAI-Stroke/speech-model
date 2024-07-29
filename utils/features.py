from typing import Dict
import librosa
import numpy as np

from parselmouth import praat
from .object import AudioObject


def get_statistics(array):
    if array.size > 0 :
         return {
            'max':np.max(array),
            'min':np.min(array),
            'mean':np.mean(array),
            'median':np.median(array),
            'std':np.std(array),
            'q_25':np.percentile(array, 25),
            'q_75':np.percentile(array, 75),
        }
    else:
         return {
            'max':None,
            'min':None,
            'mean':None,
            'median':None,
            'std':None,
            'q_25':None,
            'q_75':None,
        }


def extract_energy(audio:AudioObject) -> np.float32:
    return np.sum(audio.y**2)

def extract_mfcc(audio) -> np.ndarray:
    return librosa.feature.mfcc(y=audio.y, sr=audio.sr)


def extract_f0(audio:AudioObject, 
               fmin=librosa.note_to_hz('C2'), 
               fmax=librosa.note_to_hz('C7'))->dict:
    f0, voiced_flag, voiced_probs = librosa.pyin(audio.y, 
                                                fmin=fmin, 
                                                fmax=fmax)
    f0_voiced = f0[~np.isnan(f0)]
    feature_dict = get_statistics(f0_voiced)
    feature_dict['f0'] = f0
    return feature_dict

def extract_jitter(audio:AudioObject, 
                   time_range_l=0.0, 
                   time_range_r=0.0, 
                   period_floor=0.0001, 
                   period_ceiling=0.02, 
                   maximum_period_factor=1.3):
    """
    Jitter: 단위 시간 안에서 발음의 성대 진동(피치)의 변화를 나타냄.

    Parameters:
    - audio (AudioObject): 분석할 오디오 데이터를 포함하는 AudioObject.
    - time_range_l (float): 분석을 시작할 시간 (초). 기본값은 0.0으로 오디오의 시작을 의미.
    - time_range_r (float): 분석을 종료할 시간 (초). 기본값은 0.0으로 오디오의 끝을 의미.
    - period_floor (float): 피치 기간의 최소값 (초). 기본값은 0.0001초.
    - period_ceiling (float): 피치 기간의 최대값 (초). 기본값은 0.02초.
    - maximum_period_factor (float): 최대 기간 요인. 피치 기간의 변동성을 제어하며, 기본값은 1.3.

    Returns:
    - jitter (float): 주어진 시간 범위 내에서의 jitter 값.
    """
    sound  = audio.snd
    point_process = praat.call(sound, "To PointProcess (periodic, cc)", 75, 625)
    jitter = praat.call(point_process, "Get jitter (local)", 
                        time_range_l, 
                        time_range_r,
                        period_floor,
                        period_ceiling,
                        maximum_period_factor
                        ) 
    return jitter

def extract_shimmer(audio:AudioObject,
                    time_range_l = 0.0,
                    time_range_r = 0.0,
                    shortest_period = 0.0001,
                    longest_period=0.02,
                    maximum_period_factor=1.3,
                    maximum_amplitude_factor=1.6):
    """
    Shimmer: 음성파형에서 각 지점의 진폭 값의 변화가 얼마나 규칙적인지 나타냄.

    Parameters:
    - audio (AudioObject): 분석할 오디오 데이터를 포함하는 AudioObject.
    - time_range_l (float): 분석을 시작할 시간 (초). 기본값은 0.0으로 오디오의 시작을 의미.
    - time_range_r (float): 분석을 종료할 시간 (초). 기본값은 0.0으로 오디오의 끝을 의미.
    - shortest_period (float): 분석할 피치 기간의 최소값 (초). 기본값은 0.0001초.
    - longest_period (float): 분석할 피치 기간의 최대값 (초). 기본값은 0.02초.
    - maximum_period_factor (float): 피치 기간의 최대 요인. 기본값은 1.3.
    - maximum_amplitude_factor (float): 진폭의 최대 요인. 기본값은 1.6.

    Returns:
    - shimmer (float): 주어진 시간 범위 내에서의 shimmer 값.
    """
    sound = audio.snd
    point_process = praat.call(sound, "To PointProcess (periodic, cc)", 75, 625)
    shimmer = praat.call([sound, point_process], 
                        "Get shimmer (local)",
                        time_range_l, 
                        time_range_r, 
                        shortest_period,
                        longest_period,
                        maximum_period_factor,
                        maximum_amplitude_factor
                        )
    return shimmer

def extract_hnr(audio: AudioObject,
                time_step: float = 0.01,
                minimum_pitch: float = 75.0,
                silence_threshold: float = 0.1,
                periods_per_window: float = 1.0,
                time_range_l: float = 0.0,
                time_range_r: float = 0.0):
    """
    HNR (Harmonics-to-Noise Ratio): 음성 신호의 주기적인 성분과 비주기적인 성분(노이즈)의 비율을 나타냄.

    Parameters:
    - audio (AudioObject): 분석할 오디오 데이터를 포함하는 AudioObject.
    - time_step (float): 분석에서 사용할 시간 간격(초). 기본값은 0.01초.
    - minimum_pitch (float): 분석할 최소 피치(Hz). 기본값은 75.0Hz.
    - silence_threshold (float): 무음으로 간주할 진폭 임계값. 기본값은 0.1.
    - periods_per_window (float): window당 주기 수. 기본값은 1.0.
    - time_range_l (float): 분석을 시작할 시간 범위(초). 기본값은 0.0초로 오디오의 시작을 의미.
    - time_range_r (float): 분석을 종료할 시간 범위(초). 기본값은 0.0초로 오디오의 끝을 의미.

    Returns:
    - hnr (float): 주어진 시간 범위 내에서의 HNR 값.
    """
    sound = audio.snd
    harmonicity = praat.call(sound, "To Harmonicity (cc)", 
                       time_step,
                       minimum_pitch,
                       silence_threshold,
                       periods_per_window)
    hnr = praat.call(harmonicity, "Get mean", 
                        time_range_l, 
                        time_range_r)
    return hnr


def extract_formant_freq(audio:AudioObject,
                         time_step = 0.01,
                         max_number_of_formants = 5,
                         maximum_formant = 5500, #  (= adult female)
                         window_length = 0.025,
                         pre_emphasis_from = 50
                         )-> Dict[str, dict]:
    """
    Formant Frequency를 추출합니다.
    
    Parameters:
    - audio (AudioObject): 분석할 오디오 데이터를 포함하는 AudioObject.
    - time_step (float): 몇 초의 시간 간격으로 포먼트 값을 구할 것인지. 기본값은 0.01초.
        - 작게 할 수록 시간 축에 대해 촘촘하고 해상도가 높은 그림을 보여줌.
    - max_number_of_formants (int): 최대 포먼트 갯수. 기본값은 5.
        - 대체로 2-6까지의 숫자 가운데 하나를 넣으면 됨.
        - 이 값을 높일 수록 포먼트 갯수가 더 많이 찾아지지만, 음성의 특성을 적절히 나타내는 값으로 보기는 어렵다.
    - maximum_formant (float): 포먼트 분석할 범위의 최대 값. 기본값은 5500Hz (성인 여성의 경우).
    - window_length (float): 분석할 창의 지속시간. 기본값은 0.025초.
    - pre_emphasis_from (float): 입력한 주파수 지점부터 최대 포먼트 값까지 서서히 스펙트럼의 진폭값 증가시킴. 기본값은 50Hz.

    Returns:
    - feature_dict (Dict[str, np.ndarray]): 각 포먼트 주파수와 그 통계치를 포함하는 딕셔너리.
    """
    # http://blog.syntheticspeech.de/2021/03/10/how-to-extract-formant-tracks-with-praat-and-python/

    sound = audio.snd
    pointProcess = praat.call(sound, 'To PointProcess (periodic, cc)', 75, 625)
    num_points = praat.call(pointProcess, 'Get number of points')
    formants = praat.call(sound, "To Formant (burg)",
                         time_step, 
                           max_number_of_formants, 
                       maximum_formant,
                         window_length,
                           pre_emphasis_from)
    
    formant_lists = [[] for _ in range(max_number_of_formants)]
    for point in range(0, num_points):
        point += 1
        time = praat.call(pointProcess, 'Get time from index', point)
        for i in range(max_number_of_formants):
            formant_value =  praat.call(formants, "Get value at time", i+1, time, "Hertz", "Linear")
            formant_lists[i].append(formant_value)

    feature_dict = dict()
    for i, formant_list in enumerate(formant_lists):
        f_name = f'f{i+1}'
        feature_dict[f'{f_name}_dict'] =  get_statistics(np.array(formant_list)) 
        feature_dict[f'{f_name}_dict'][f_name] = np.array(formant_list)


    return feature_dict