import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn import clone
import pandas as pd
import os
import copy
import random
import numpy as np

warnings.filterwarnings('ignore')
#   Normalized data storage directory
STANDARD_DIR = 'standard'
#   Results Save directory
RESULT_DIR = 'result'
#   Accuracy Saving directory
RESULT_FILE = 'cv_accuracy.csv'
#   Five-fold 
N_SPLITS = 5
#   Random seeds
SEED = 20
random.seed(SEED)

#   Using the MinMaxScaler to normalize datasets
def origin2standard(filename):
    if os.path.exists(filename) == False:
        raise FileNotFoundError(f'{filename}不存在！')
    target_path = f'{STANDARD_DIR}/{os.path.split(filename)[1]}'
    if os.path.exists(STANDARD_DIR)==False:
        os.mkdir(STANDARD_DIR)
    if os.path.exists(target_path):
        print(target_path, " 已经存在!")
        return
    data = pd.read_csv(filename, header=None, index_col=0)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    idx = X.index
    X = X.astype(np.float64)
    min_max_scaler = MinMaxScaler()
    X = min_max_scaler.fit_transform(X)
    X = pd.DataFrame(X, index=idx)
    df = pd.concat([X, y], axis=1)
    df.to_csv(target_path, header=0)
    print(f'{filename}归一化后，写入{target_path}')

#   Read datasets
def readDataFile(filename):
    if os.path.exists(filename) == False:
        raise FileNotFoundError(f'{filename}不存在！')
    data = pd.read_csv(filename, header=None, index_col=0)
    print(f'之前：{len(data)}')
    data = preprocess(data)
    print(f'去掉不一致之后：{len(data)}')
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    y = y+1
    label = y.unique()
    print(f'样本数:{len(X)},特征数:{X.shape[1]},标记数:{len(label)} {label}')

    if np.any(label == 0):
        y[y==0] = label.max()+1
        print(f'将标记中0替换为{label.max()+1}')
    return X, y

#   Remove unconsistent samples
def removeUnconsistent(group):
    if len(group)==1:
        return group
    elif len(group.iloc[:,-1].unique())==1:
        return group

#   Preprocessing 
def preprocess(data):
    # print(data.columns.tolist()[:-1])
    data = data.groupby(data.columns.tolist()[:-1],as_index=False).apply(removeUnconsistent)
    # print(data.index)
    if isinstance(data.index,pd.MultiIndex):
        data.index = [b for a,b in data.index]
    return data.sort_index()

#   Get the splitted data
def getCVSplit(dataList):
    for c in dataList:
        path = f'../standard/{c}.csv'
        print('5折交叉数据划分，开始处理 ', path)
        X, y = readDataFile(path)
        sk = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        i = 1
        d = {i: [] for i in range(1,6)}
        for train_idx, valid_idx in sk.split(X, y):
            d[i].append(train_idx)
            d[i].append(valid_idx)
            i += 1
        import numpy as np
        np.save(f'{c}.npy', d)

def list_equal(list1: list, list2: list) -> bool:
    list_result = []
    len1 = len(list1)
    len2 = len(list2)
    if len1 == len2:
        for i in list1:
            if i in list2 and list1.count(i) == list2.count(i):
                list_result.append(True)
            else:
                list_result.append(False)
        if False in list_result:
            return False
        else:
            return True
    else:
        return  False





def getCVAccuracy_noise(dataList,type, base_estimator,name = True, noise=0, **param):
    print(type)
    for c in dataList:
        d = []
        if name:
            param['name'] = c
        d1 = [c,type, param, noise]
        path = f'../standard/{c}.csv'
        print('5折交叉,计算时间/准确率，开始处理 ', path)
        X, y = readDataFile(path)
        sk = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        scores = []
        print(param)
        i = 1
        for train_idx, valid_idx in sk.split(X, y):
            print(f'第{i}次：')
            i += 1
            clf = clone(base_estimator)
            clf.set_params(**param)
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            if noise > 0:
                y_train_copy = copy.deepcopy(y_train)
                idxs = y_train_copy.index.tolist()
                size = int(len(y_train_copy) * noise)
                random_index = np.array(random.sample(idxs, size))

                classes = np.unique(y_train_copy)
                for idx in random_index:
                    L_ = list(classes)
                    L_.remove(y_train_copy.loc[idx])
                    new_class = random.choice(L_)
                    y_train_copy.loc[idx] = new_class
                clf.fit(X_train, y_train_copy)
            else:
                clf.fit(X_train, y_train)
            scores.append(clf.score(X.iloc[valid_idx], y.iloc[valid_idx]))
            print(f'准确率：{scores[-1]}')
        d1.extend(scores)
        scores = np.array(scores)
        d1.append(scores.mean())
        print(f'平均准确率：{d1[-1]}')
        d1.append(scores.std())
        print(f'平均标准差：{d1[-1]}')
        d.append(d1)
        df_acc = pd.DataFrame(d,
                              columns=['data', 'algorithm', 'param', 'noise', '1', '2', '3', '4', '5', 'mean', 'std'])
        if os.path.exists(RESULT_DIR)==False:
            os.makedirs(RESULT_DIR)
        if os.path.exists(f"{RESULT_DIR}/{RESULT_FILE}"):
            df_accuracy = pd.read_csv(f"{RESULT_DIR}/{RESULT_FILE}")
            df_concat = pd.concat([df_accuracy, df_acc])
            df_concat.to_csv(f"{RESULT_DIR}/{RESULT_FILE}", index=False)
        else:
            df_acc.to_csv(f"{RESULT_DIR}/{RESULT_FILE}", index=False)



