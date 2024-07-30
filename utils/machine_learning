import pandas as pd
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def perform_LR(X_train, y_train):
    # 7. LR에서 활용할 hyperparameter 선정하기
    param_grid_lr = [
        {
            'classifier__penalty': ['l2', 'none'],
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'classifier__solver': ['lbfgs', 'sag'],
            'classifier__max_iter': [100, 200, 500, 1000, 2000],
        },
        {
            'classifier__penalty': ['l1', 'l2'],
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'classifier__solver': ['liblinear'],
            'classifier__max_iter': [100, 200, 500, 1000, 2000],
        },
        {
            'classifier__penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'classifier__solver': ['saga'],
            'classifier__max_iter': [100, 200, 500, 1000, 2000],
            'classifier__l1_ratio': [0.5],
        }
    ]

    pipeline_lr = Pipeline(
        steps=[('smote', SMOTE(random_state=42)), ('classifier', LogisticRegression(random_state=42))])
    grid_search_lr = GridSearchCV(estimator=pipeline_lr, param_grid=param_grid_lr,
                                  scoring=make_scorer(f1_score, average='binary'), cv=StratifiedKFold(5), n_jobs=-1,
                                  verbose=0)
    grid_search_lr.fit(X_train, y_train)

    print("Best hyperparameters for Logistic Regression: ", grid_search_lr.best_params_)

    return grid_search_lr.best_estimator_


def perform_SVM(X_train, y_train):
    # 7. SVM에서 활용할 hyperparameter 선정하기
    param_grid_svm = {
        'classifier__C': [0.01, 0.05, 0.1, 0.5, 1],
        'classifier__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'classifier__gamma': ['scale', 'auto'],
    }

    # 8. 데이터 oversampling 및 모델 학습 시행
    pipeline_svm = Pipeline(steps=[('smote', SMOTE(random_state=42)), ('classifier', SVC(probability=True, random_state=42))])
    grid_search_svm = GridSearchCV(estimator=pipeline_svm, param_grid=param_grid_svm, scoring=make_scorer(f1_score, average='binary'), cv=StratifiedKFold(5), n_jobs=-1, verbose=0)
    grid_search_svm.fit(X_train, y_train)

    # 9. 최적의 결과를 도출하는 Hyperparameter 출력하기
    print("Best hyperparameters for SVM: ", grid_search_svm.best_params_)

    # 10. 최적화 모델 기준 Performance 출력하기
    return grid_search_svm.best_estimator_


def perform_RF(X_train, y_train):
    # 7. SVM에서 활용할 hyperparameter 선정하기
    param_grid_rf = {
        'classifier__n_estimators': [100, 200, 500],
        'classifier__max_depth': [10, 20, 30],
        'classifier__min_samples_split': [5, 10],
        'classifier__min_samples_leaf': [2, 4],
    }

    pipeline_rf = Pipeline(steps=[('smote', SMOTE(random_state=42)), ('classifier', RandomForestClassifier(random_state=42))])
    grid_search_rf = GridSearchCV(estimator=pipeline_rf, param_grid=param_grid_rf, scoring=make_scorer(f1_score, average='binary'), cv=StratifiedKFold(5), n_jobs=-1, verbose=0)
    grid_search_rf.fit(X_train, y_train)

    print("Best hyperparameters for Random Forest: ", grid_search_rf.best_params_)

    return grid_search_rf.best_estimator_


# 'stroke' 변수를 제외한 모든 변수를 특성으로 사용
df_preprocessed = pd.read_csv('../preprocess/another_preprocessed_data.csv')
df_preprocessed = df_preprocessed.drop(columns='fname')

X = df_preprocessed.drop('stroke', axis=1)
y = df_preprocessed['stroke']
print('뇌졸중 기준을 만족하는 사람의 수:', sum(y))

# 5. Training and Testing Set 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

log_reg = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
log_reg.fit(X_train, y_train)

# 6. Performance metrics 추출을 위한 함수 정의하기
def get_performance_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        'ROC-AUC': round(float(roc_auc_score(y_test, y_proba)), 5),
        'Accuracy': round(float(accuracy_score(y_test, y_pred)), 5),
        'Recall': round(float(recall_score(y_test, y_pred)), 5),
        'Precision': round(float(precision_score(y_test, y_pred)), 5),
        'F1 Score': round(float(f1_score(y_test, y_pred)), 5)
    }


# 7. 각 모델에 대하여 학습 실행하기
model_lr = perform_LR(X_train, y_train)
model_svm = perform_SVM(X_train, y_train)
model_rf = perform_RF(X_train, y_train)

# 8. 각 모델에 대하여 평가 데이터 구축하기
metrics = {}

model_train_name = ['LR_train', 'SVM_train', 'RF_train']
model_test_name = ['LR_test', 'SVM_test', 'RF_test']
models = [model_lr, model_svm, model_rf]
for idx in range(3):
    metrics[model_train_name[idx]] = get_performance_metrics(models[idx], X_train, y_train)
    metrics[model_test_name[idx]] = get_performance_metrics(models[idx], X_test, y_test)

metrics_df = pd.DataFrame(metrics).T
metrics_df = metrics_df[['ROC-AUC', 'Accuracy', 'Recall', 'Precision', 'F1 Score']]  # 열 순서 정렬
metrics_df.to_csv('../model_comparison/performance_metrics_1.csv')

print(metrics_df)
