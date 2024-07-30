import pandas as pd
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', None)

# 1. 데이터 프레임 구축
stroke = pd.read_csv('stroke_sample.csv', index_col=0)
non_stroke = pd.read_csv('non_stroke_sample.csv', index_col=0)
df = pd.concat([stroke, non_stroke], ignore_index=True)

# 'label' 열의 값을 정수로 변환
df['label'] = df['label'].fillna('non_stroke')
df['label'] = df['label'].map({'stroke': 1, 'non-stroke': 0})
df['stroke'] = df['label'].astype(int)

to_delete = ['age', 'chunk_i', 'dirname', 'f_id', 'p_id', 'sex', 'label']
df = df.drop(columns=to_delete)
df = df.dropna(how='any')
print(df)

variables = df.columns.tolist()
print(variables)

# 2. 데이터 상태 확인
# 모든 열의 이름, non-null count, 자료형을 원하는 형식으로 출력
info = df.dtypes.reset_index()
info.columns = ['Column', 'Dtype']
info['Non-Null Count'] = df.notnull().sum().values
info = info[['Column', 'Non-Null Count', 'Dtype']]

# 출력
print(info)

# 3. 스케일링 시행
processed_data = []

# 스케일러 초기화
scaler = StandardScaler()

exception = ['stroke', 'fname']
# all_variables 리스트를 순회하며 변수 처리
for var in variables:
    if var not in exception:
        # 연속형 변수 스케일링
        scaled_data = scaler.fit_transform(df[[var]])
        processed_data.append(pd.DataFrame(scaled_data, columns=[var], index=df.index))
    else:
        processed_data.append(df[[var]])

# 새로운 데이터프레임 생성
df_preprocessed = pd.concat(processed_data, axis=1)

# 결과 확인
print(df_preprocessed.head())
df_preprocessed.to_csv('preprocessed_data.csv', index=False)
