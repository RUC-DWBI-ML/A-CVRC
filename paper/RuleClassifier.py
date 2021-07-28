
__Filename__ = 'RuleClassifier'
__Author__ = 'Peng Ni'
__Date__ = '2020/5/7'
import pandas as pd
import numpy as np
from paper.AttributeValueReduction import AttributeValueReduction,time_count

import copy,os,time
import threading

__all__ = ['RuleClassifier']

class RuleClassifier(object):
    def __init__(self, alpha=0, mode=1, name='test', verbose=0, save = False):
        #   Dataset name
        self.name = name
        #   Threshold value
        self.alpha = alpha
        #   mode=1 or mode =2
        self.mode = mode
        #   Wether saving information
        self.save = save
        #   Final Rule sets
        self.minRules = {}
        #   Wether to print information(0 or 1)
        self.verbose = verbose
        #   All attribute value reduction
        self.allRules = {}
        #   Cover list
        self.coverList = {}

    def get_params(self, deep=False):
        return {'alpha': self.alpha,'name':self.name,'verbose':self.verbose,'mode':self.mode,'save':self.save}

    def set_params(self, **params):
        if not params:
            return self
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def score(self, X, y):
        current_decision = self.predict(X)
        accuracy = sum(current_decision == y) / len(y)
        return accuracy

    def predict(self, X):
        rule_num = len(self.minRules)
        row = len(X)
        deci_rule_set = self.y_train.loc[self.minRules.keys()].values
        cover_object = np.zeros((row, rule_num))
        for i, (k, idx) in enumerate(self.minRules.items()):
            consistence = np.array([self.attributeValueReduction.getCons(k)] * row).reshape((-1, 1))
            attribute = X.loc[:, idx].values
            rule_part = self.X_train.loc[k, idx]
            simi = (np.min(self.relation_MIN(np.tile(rule_part, (row, 1)), attribute), axis=1)).reshape((-1, 1))
            alpha_array = np.array([self.alpha] * row).reshape((-1, 1))
            D1 = (self.S_conorm_TL(1 - simi, alpha_array) < consistence) * 1
            D2 = abs(self.S_conorm_TL(1 - simi, alpha_array) - consistence) < 1e-6
            D1[D2] = 0
            cover_object[:, i] = D1[:, 0]
        t = (cover_object * np.tile(deci_rule_set, (row, 1))).astype(np.int32)
        current_decision = [np.argmax(np.bincount(np.array(t[i, :])[np.flatnonzero(t[i, :])])) if len(
            np.flatnonzero(t[i, :])) > 0 else 0 for i in range(row)]
        xxx = np.where(np.array(current_decision) == 0)[0]
        s = len(xxx)
        if s > 0:
            dis_all = np.zeros((s, rule_num))
            for i, (k, idx) in enumerate(self.minRules.items()):
                attribute = X.iloc[xxx, np.array(idx) - 1].values
                rule_part = self.X_train.loc[k, idx]
                simi = np.min(self.relation_MIN(np.tile(rule_part, (s, 1)), attribute), axis=1)
                dis_all[:, i] = simi
            cover_position = np.argmin(dis_all, axis=1)
            deci_rule_set = deci_rule_set[cover_position]
            for k, index in enumerate(xxx):
                current_decision[index] = deci_rule_set[k]
        return current_decision
    def S_conorm_TL(self,x, y):
        rowx, colx = x.shape
        rowy, coly = y.shape
        if rowx != rowy or colx != coly:
            raise ValueError('the input arguments are not consistent')
        result = np.minimum(1, x + y)
        result[abs(x + y - 1) < 1e-6] = 1
        return result
    def TL_implicator(self,a, b):
        rowa, cola = a.shape
        rowb, colb = b.shape
        if rowa != rowb or cola != colb:
            raise ValueError('the input arguments are not consistent')
        result = np.minimum(1, 1 - a + b)
        result[abs(-a + b) < 1e-6] = 1
        return result
    def relation_MIN(self,x,y):
        rowx, colx = x.shape
        rowy, coly = y.shape
        if rowx != rowy or colx != coly:
            raise ValueError('the input arguments are not consistent')
        return self.TL_implicator(np.maximum(x,y),np.minimum(x,y))

    @time_count
    def reductionRule(self):
        if self.verbose > 0:
            print('当前执行线程:', threading.current_thread().getName())
        idx = copy.deepcopy(self.idx)
        dd = {}
        for x in idx:
            coverDegree = self.calCoverDegree(x)
            if dd.get(coverDegree) is None:
                dd[coverDegree] = [x]
            else:
                dd[coverDegree].append(x)
        k = max(dd.keys())
        kk = min(dd[k])
        if self.verbose > 0:
            print(f'选择 {kk},覆盖度:{k}')
        self.minRules[kk] = self.allRules[kk]
        A = self.coverList[kk]
        idx = idx ^ A
        while len(idx) > 0:
            dd = {}
            for x in idx:  # desc设置名称,ncols设置进度条长度.postfix以字典形式传入详细信息
                coverDegree = self.updateCoverDegree(x, A)
                if dd.get(coverDegree) is None:
                    dd[coverDegree] = [x]
                else:
                    dd[coverDegree].append(x)
            k = max(dd.keys())
            kk = min(dd[k])
            if self.verbose > 0:
                print(f'选择 {kk},覆盖度:{k}')
            self.minRules[kk] = self.allRules[kk]
            A = self.coverList[kk]
            idx = idx ^ A

    def updateCoverDegree(self, x, A):
        self.coverList[x] = self.coverList[x] ^ (self.coverList[x] & A)
        return len(self.coverList[x])

    def calCoverDegree(self, x):
        y = self.y_train.loc[x]
        abc = self.y_train == y
        attr = self.allRules[x]
        disMatrix = abs(self.X_train.loc[x, attr] - self.X_train.loc[abc, attr])
        cons = disMatrix.max(axis=1)
        cover = cons < self.attributeValueReduction.getCons(x)
        coverDegree = cover.sum()
        self.coverList[x] = cover[cover == True].index
        return coverDegree

    @time_count
    def reductionRule_0(self):
        self.X_train_copy = self.X_train.copy()
        self.y_train_copy = self.y_train.copy()
        idx = self.X_train_copy.index
        while len(idx) > 0:
            dd = {}
            for x in idx:  
                coverDegree = self.calCoverDegree_0(x)
                if dd.get(coverDegree) is None:
                    dd[coverDegree] = [x]
                else:
                    dd[coverDegree].append(x)
            k = max(dd.keys())
            kk = min(dd[k])
            if self.verbose > 0:
                print(f'选择 {kk},覆盖度:{k}')
            self.X_train_copy.drop(self.coverList[kk], inplace=True)
            self.y_train_copy.drop(self.coverList[kk], inplace=True)
            self.minRules[kk] = self.allRules[kk]
            idx = self.X_train_copy.index

    def calCoverDegree_0(self, x):
        y = self.y_train_copy.loc[x]
        abc = self.y_train_copy == y
        attr = self.allRules[x]
        disMatrix = abs(self.X_train_copy.loc[x, attr] - self.X_train_copy.loc[abc, attr])
        cons = disMatrix.max(axis=1)
        cover = cons < self.attributeValueReduction.getCons(x)
        coverDegree = cover.sum()
        self.coverList[x] = cover[cover == True].index
        return coverDegree

    def fit(self, X_train, y_train):
        if isinstance(X_train, np.ndarray):
            col = range(1,X_train.shape[1]+1)
            row = range(1,X_train.shape[0]+1)
            self.X_train = pd.DataFrame(X_train,columns=col,index=row)
            self.y_train = pd.Series(y_train,index=row)
        else:
            self.X_train = X_train
            self.y_train = y_train
        self.X_train.index = self.X_train.index.astype(str)
        self.y_train.index = self.y_train.index.astype(str)
        self.idx = self.X_train.index
        self.data2excel = [self.name, len(self.idx), self.X_train.shape[1], len(self.y_train.unique()),
                           self.alpha]
        self.data2excel.append(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        start = time.time()
        self.attributeValueReduction = AttributeValueReduction(alpha=self.alpha, name=self.name, mode=self.mode,save=self.save,
                                                               verbose=self.verbose)
        self.attributeValueReduction.fit(self.X_train, self.y_train)
        self.allRules = self.attributeValueReduction.getReduct()
        if self.mode == 1:
            self.reductionRule()
        elif self.mode == 2:
            self.reductionRule_0()
        end = time.time()
        self.data2excel.append(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        self.data2excel.append(end - start)
        if self.save:
            self.save2excel()

    def save2excel(self, path=None):
        index_s = self.minRules.keys()
        self.data2excel.append(len(index_s))
        a = []
        for key, value in self.minRules.items():
            for i in value:
                if i not in a:
                    a.append(i)
        print(f'选择特征数目={len(a)}')
        self.data2excel.append(len(a))
        df = pd.DataFrame([self.data2excel],
                          columns=["数据集", "训练样本数目", "特征数目", "标记数目", "模型阈值", "开始时间", "结束时间", "总时间",
                                   "规则集大小","选择特征数目"])
        if path is None:
            path = f"result/result_RC_{self.mode}.csv"
        dirname, basename = os.path.split(path)
        if os.path.exists(dirname) == False and len(dirname.strip()) > 0:
            os.makedirs(dirname)
        if os.path.exists(path):
            df_old = pd.read_csv(path)
            df_concat = pd.concat([df_old, df], ignore_index=True)
            df_concat.to_csv(path, index=False, encoding="utf_8_sig")
        else:
            df.to_csv(path, index=False, encoding="utf_8_sig")
        if self.verbose > 0:
            print(f'时间结果保存到{path}')

if __name__ == '__main__':
    name = 'test'
    dataPath = f'../standard/{name}.csv'
    from paper.common import readDataFile
    X, y = readDataFile(dataPath)
    obj = RuleClassifier(alpha=0, name=name,mode=1, verbose=0,save=True)
    obj.fit(X, y)
    print(obj.score(X, y))
