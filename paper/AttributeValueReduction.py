#   Incremental Rule-based Classifier
__Author__ = 'Peng Ni'
#   Date: 2020/4/22
import pandas as pd

import copy
import threading
import time
import os,json

__all__ = ['AttributeValueReduction','time_count']

def time_count(func):
    def inner(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        return (end - start)
    return inner

class AttributeValueReduction(object):
    def __init__(self, alpha=0, mode=1, name='test', verbose=0, save=False):
        #  Dataset name
        self.name = name
        #  Threshold value
        self.alpha = alpha
        #  mode=1 or mode =2
        self.mode = mode
        #  Wether to save the record information (True or False)
        self.save = save
        #  Wether to print information(0 or 1)
        self.verbose = verbose
        #  Consistent degree value
        self.consDegree = {}
        #  The distance of single attribute set
        self.dis_a = None
        #  The distance of current attribute subsets
        self.dis_B = None
        #  Index of heteorgeneous class
        self.xxx = None
        #  All attritute value reduction
        self.allReduct = {}

    def removeRedundancy(self, x):
        attr = self.allReduct[x]
        if len(attr) == 1:
            return
        B = copy.deepcopy(attr)

        for a in attr:
            B.remove(a)
            if len(B) == 0 or self.calConsistenceDegree(x, B) < self.consDegree[x]:
                B.append(a)
        self.allReduct[x] = B


    def calConsDegree(self):
        if self.dis_B is None:
            attr_idx = self.dis_a.min().idxmax()
            self.dis_B = self.dis_a.loc[:, attr_idx]
        else:
            lst = {c: pd.concat([self.dis_B, self.dis_a.loc[:, c]], axis=1).max(axis=1).min() for c in
                   self.dis_a.columns}
            attr_idx = pd.Series(lst).idxmax()
            self.dis_B = pd.concat([self.dis_B, self.dis_a.loc[:, attr_idx]], axis=1).max(axis=1)
        self.dis_a.drop(attr_idx, axis=1, inplace=True)
        self.xxx = self.xxx ^ self.dis_B.loc[((self.dis_B + self.alpha) >= 1)].index

        return min(self.dis_B.min() + self.alpha, 1), attr_idx

    def calConsistenceDegree(self, x, attr=None):
        if attr is None:
            disMatrix = abs(self.X_train.loc[x, :] - self.X_train.loc[self.xxx, :])
        else:
            disMatrix = abs(self.X_train.loc[x, attr] - self.X_train.loc[self.xxx, attr])
        cons = min(disMatrix.max(axis=1).min() + self.alpha, 1)
        return cons

    @time_count
    def calReductAll(self):
        if self.verbose > 0:
            print('????????????:', threading.current_thread().getName())
        if self.mode == 1:
            for x in self.idx:
                self.setXXX(x)
                self.dis_a = abs(self.X_train.loc[x] - self.X_train.loc[self.xxx])

                cons = min(self.dis_a.max(axis=1).min() + self.alpha, 1)
                if cons == 0:
                    raise ValueError(f'{x}???????????????0,????????????????????????????????????????????????')
                self.consDegree[x] = cons
                rule = self.calReduct(x)
                self.allReduct[x] = rule

                self.dis_a = None
                self.dis_B = None
                self.setXXX(x)
                self.removeRedundancy(x)
                self.xxx = None

        elif self.mode == 2:
            for x in self.idx:
                self.setXXX(x)
                cons = self.calConsistenceDegree(x)
                self.consDegree[x] = cons
                rule = self.calReduct_0(x)
                self.allReduct[x] = rule
                self.setXXX(x)
                self.removeRedundancy(x)
        else:
            raise ValueError('mode ?????????!')

    def calReduct_0(self, x):
        currConsDegree = 0
        leaf = self.X_train.columns.tolist()
        red = []
        initConsDegree = self.consDegree[x]
        while initConsDegree > currConsDegree:
            k = -1
            tmp = 0
            for i in leaf:
                red.append(i)
                tmp2 = self.calConsistenceDegree(x, red)
                if tmp2 > tmp:
                    k = i
                    tmp = tmp2
                red.remove(i)
            if k == -1:
                k = leaf[0]
            currConsDegree = tmp
            red.append(k)
            leaf.remove(k)
        return red

    def setXXX(self, x):
        y = self.y_train.loc[x]
        self.xxx = self.X_train.loc[~(self.y_train == y)].index

    def calReduct(self, x):
        currConsDegree = 0
        red = []
        initConsDegree = self.consDegree[x]
        while initConsDegree > currConsDegree:
            currConsDegree, attr_idx = self.calConsDegree()
            red.append(int(attr_idx))
        return red

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        # ????????????
        self.X_train.index = self.X_train.index.astype(str)
        self.y_train.index = self.y_train.index.astype(str)
        self.idx = self.X_train.index
        self.data2excel = {'?????????': self.name, '??????': self.alpha, '??????': '??????' if self.mode == 1 else '??????'}

        self.data2excel['?????????????????????'] = self.calReductAll()
        if self.save:
            self.save2json()
            self.save2excel()

    def getReduct(self):
        return self.allReduct

    def getCons(self, k):
        return self.consDegree[k]

    def save2json(self, path=None):
        data = {'reduct': self.allReduct, 'data': self.X_train.values.tolist(),
                'label': self.y_train.values.tolist(), 'consDegree': self.consDegree}
        if path is None:
            path = f'result/{self.name}/{self.alpha}/reduct_{self.mode}.json'
        dirname, basename = os.path.split(path)
        if os.path.exists(dirname) == False and len(dirname.strip()) > 0:
            os.makedirs(dirname)
        if self.verbose > 0:
            print(f'??????????????????????????????{path}')
        with open(path, 'w') as fw:
            json.dump(data, fw)

    def save2excel(self, path=None):
        df = pd.DataFrame(self.data2excel, index=[0])
        if path is None:
            path = f'result/{self.name}/{self.alpha}/reduct_{self.mode}.csv'
        dirname, basename = os.path.split(path)
        if os.path.exists(dirname) == False and len(dirname.strip()) > 0:
            os.makedirs(dirname)
        df.to_csv(path, index=False)
        if self.verbose > 0:
            print(f'?????????????????????{path}')

if __name__ == '__main__':
    name = 'test'
    dataPath = f'../standard/{name}.csv'
    from paper.common import readDataFile
    X, y = readDataFile(dataPath)
    for m in [1, 2]:
        obj = AttributeValueReduction(alpha=0, mode=m, name=name, verbose=0,save=False)
        obj.fit(X, y)
        break
