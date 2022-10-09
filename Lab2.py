from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler, StandardScaler, Normalizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.mixture import GaussianMixture
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
data = pd.read_csv("housing.csv")
raw = data.copy()
data.drop(columns=['median_house_value'], inplace=True)
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)
dataset = pd.DataFrame(
    {"households_population": data['households'] / data['population'], "median_income": data['median_income']})
# Encoder definition
onehot = OneHotEncoder
label = LabelEncoder
encoder_list = [label, onehot]

# Scaler definition
mm_scaler = MinMaxScaler
st_scaler = StandardScaler
ma_scaler = MaxAbsScaler
rb_scaler = RobustScaler
nm_scaler = Normalizer
scaler_list = [mm_scaler, st_scaler, ma_scaler, rb_scaler, nm_scaler]

# Model definition
kmeans = KMeans
meanshift = MeanShift
dbscan = DBSCAN
gmm = GaussianMixture
model_list = [kmeans, meanshift, dbscan, gmm]


# Check each data's type and encoding if categorical dataset
# parameter : {encoder : encoder(label, onehot....), data : data(DataFrame)}
def encoding(encoder, data):
    for x, y in zip(data, data.dtypes):
        if (y == object):
            # LabelEncoder
            if (encoder.__name__ == 'LabelEncoder'):
                encoded = encoder().fit_transform(data[x])
                data.drop(columns=[x], inplace=True)
                result = pd.DataFrame(encoded, columns=['encoded' + x])
                data = pd.concat([data, result], axis=1)

            # OnehotEncoding
            elif (encoder.__name__ == 'OneHotEncoder'):
                encoded = encoder().fit_transform(data[[x]])
                data.drop(columns=[x], inplace=True)
                tmp = encoded.toarray()
                col = ['{}_{}'.format(k, x) for k in range(len(tmp[0]))]
                result = pd.DataFrame(tmp, columns=col)
                data = pd.concat([data, result], axis=1)
            else:
                print("Unknown Encoder!! \n\n")
    return data


# scaling data using get parameter
# parameter : {scaler : sacaler(Minmax, Standard....), data : data(DataFrame)}
# scaling operation
def scaling(scaler, data):
    scaled = scaler().fit_transform(data)
    scaled = pd.DataFrame(scaled, columns=data.columns)
    return scaled


# training model using get model
# parameter : {model : clustering model (Kmeans, DBSCAN.....),data : data(DataFrame), n : cluster number(필수x)}
def modeling(model, data, n):
    if model.__name__ == "KMeans":
        params = {"n_clusters": [2, 6, 8, 10, 12], "random_state": [42]}
        grid = GridSearchCV(model(), param_grid=params, verbose=3, cv=1)
        a = model(n_clusters=n).fit_predict(data)
        return a
    elif model.__name__ == "MeanShift":
        a = model().fit_predict(data)
        return a
        print("yet")
    elif model.__name__ == "DBSCAN":
        a = model().fit_predict(data)
        return a
        print("yet")
    elif model.__name__ == "GaussianMixture":
        a = model(n_components=n).fit_predict(data)
        return a
        print("yet")
    elif model.__name__ == "CLARANS":
        print("yet")
    else:
        print("Unknown Model!!\n\n")


# 결과 데이터를 시각화
def scatter(data):
    plt.scatter(x=data['households_population'], y=data['median_income'], s=1, c=data['cluster'])
    plt.ylabel("median_income")


# training using get encoder, scaler, model list and make plot
# parameter : {model_list : clustering model list [DBSCAN,KMeans....], scaler_list : [MinMax,Scaler...],
#       encoder_list : encoder list [OneHot, Label....], data : dataset (DataFrame)}
def ML(model_list, scaler_list, encoder_list, data):
    for encoder in encoder_list:
        encoded = encoding(encoder, data.copy())
        enc = encoder.__name__
        for scaler in scaler_list:
            scaled = scaling(scaler, encoded)
            plt.title(scaler.__name__)
            for model, n in zip(model_list, range(1, len(model_list) + 1)):
                tmp = scaled.copy()
                result = modeling(model, tmp, 4)
                tmp['cluster'] = result
                plt.subplot(len(model_list) + 1, 1, n)
                scatter(tmp)
                plt.xlabel(model.__name__)
                print("{} is end".format(model.__name__))
            plt.show()
            print("---------{} is end----------".format(scaler.__name__))


ML(model_list, scaler_list, encoder_list, dataset)



