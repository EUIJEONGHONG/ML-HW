import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, Normalizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Load data
data = pd.read_csv("breast-cancer-wisconsin.data")
data.columns = ['id Number',
                'Clump Thickness',
                'Uniformity of Cell Size',
                'Uniformity of Cell Shape',
                'Marginal Adhesion',
                'Single Epithelial Cell Size',
                'Bare Nuclei',
                'Bland Chromatin',
                'Normal Nucleoli',
                'Mitoses',
                'Class']

# confirm data's information
print(data.describe())
print(data.info())
print(data['Bare Nuclei'].value_counts())
# data has wrong value '?'
data.replace('?', np.nan, inplace=True)
data.dropna(inplace=True)
data.reset_index(inplace=True, drop=True)
data = data.astype('int64')
target = data[['Class']]
data = data.iloc[:, :-1]
data.drop(columns=['id Number', 'Mitoses'], inplace=True)
train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.2, shuffle=True, stratify=target,
                                                    random_state=42)

# sclaer 선언
minmax = MinMaxScaler()
zscore = StandardScaler()
maxabs = MaxAbsScaler()
normalizer = Normalizer()
robust = RobustScaler()
scaler_list = [minmax, maxabs, zscore, robust, normalizer]

# model's hypherparameter
dt_params = {'criterion': ['gini', 'entropy'], 'max_depth': [None, 2, 3, 4, 5, 6]}
svm_params = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001], 'kernel': ['rbf']}
lr_params = {}
param = [dt_params, svm_params, lr_params]

# model 선언
dt = DecisionTreeClassifier(criterion='entropy')
lr = LogisticRegression()
svm = SVC()
model_list = [dt, svm, lr]

# kfold list
kfold_list = [3, 5, 10]

# function

# parameter = data, scaler, encoder, model
est = []
sca = []
kfo = []
par = []
sco = []

for scaler in scaler_list:
    print(str(scaler))
    data = scaler.fit_transform(data)
    param = [dt_params, svm_params, lr_params]
    model_list = [dt, svm, lr]
    for model, param in zip(model_list, param):
        print(str(model), str(param))
        for k in kfold_list:
            txt = "k={}, scaler = {}, model = {}".format(k, str(scaler), str(model))
            print("\n\n", txt)
            grid = GridSearchCV(model, param_grid=param, cv=k, verbose=3)
            grid.fit(data, target)

            est.append(str(grid.best_estimator_))
            sca.append(str(scaler))
            kfo.append(k)
            par.append(str(grid.best_params_))
            sco.append(grid.best_score_)

        # 결과 저장
result = pd.DataFrame({'estimator': est, 'scaler': sca, 'Kfold': kfo, 'Parameter': par, 'score': sco})
result.to_csv("PHW1_result.csv")

