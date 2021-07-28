#  Rule-based Classifier
__Author__ = 'Peng Ni'
#   Date: 2020/6/10
import pandas as pd

import copy, time, os, json
import numpy as np
import math

__all__ = ['BatchSampleRuleClassifier']

def time_count(func):
    def inner(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        return (end - start)
    return inner

class BatchSampleRuleClassifier(object):
    def __init__(self, alpha=0, name='test', sample_percent=0.1, early=0, verbose=0,mode = 1):
        #  Dataset name
        self.name = name
        #  Threshold value
        self.alpha = alpha
        #  Threshold value
        self.verbose = verbose
        #  Consistent degree value
        self.consDegree = {}
        #  The distance of single attribute
        self.dis_a = None
        #  The distance of current attribute subsets
        self.dis_B = None
        #  Index of heteorgeneous class
        self.xxx = None
        #   Batch size percent
        self.sample_percent = sample_percent
        #   All attribute value reduction
        self.allReduct = {}
        #   Final rule sets
        self.minRules = {}
        #   Cover list
        self.coverList = {}
        #   Irrelevant rule list
        self.coverList2 = {}
        #   rule lists
        self.rule_idx = []
        #   Threshold value
        self.early = early
        # mode =1 or mode = 2
        self.mode = mode

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

    def calCD(self):
        dd = {}
        for x in self.rule_idx:
            coverDegree = self.calCoverDegree(x) + self.calCoverDegree2(x)
            if dd.get(coverDegree) is None:
                dd[coverDegree] = [x]
            else:
                dd[coverDegree].append(x)
        cd = max(dd.keys())
        kk = min(dd[cd])
        return kk, cd

    def updateCD(self, kk):
        dd = {}
        A = self.coverList[kk]
        AA = self.coverList2[kk]
        y = self.y_train.loc[kk]
        for x in self.rule_idx:  
            if self.y_train.loc[x] == y:
                coverDegree = self.updateCoverDegree(x, A, AA)
            else:
                coverDegree = len(self.coverList[x]) + len(self.coverList2[x])
            if dd.get(coverDegree) is None:
                dd[coverDegree] = [x]
            else:
                dd[coverDegree].append(x)
        cd = max(dd.keys())
        kk = min(dd[cd])
        return kk, cd

    def calCoverDegree(self, x):
        y = self.y_train.loc[x]
        abc = self.y_train.loc[self.rule_idx] == y
        abc = abc[abc == True].index
        attr = self.allReduct[x]
        disMatrix = abs(self.X_train.loc[x, attr] - self.X_train.loc[abc, attr])
        cons = disMatrix.max(axis=1)

        cover = cons < self.consDegree[x]
        coverDegree = cover.sum()

        self.coverList[x] = cover[cover == True].index
        return coverDegree

    def calCoverDegree2(self, x):
        y = self.y_train.loc[x]
        abc = self.y_train_copy == y
        abc = abc[abc == True].index
        attr = self.allReduct[x]
        disMatrix = abs(self.X_train.loc[x, attr] - self.X_train.loc[abc, attr])
        cons = disMatrix.max(axis=1)

        cover = cons < self.consDegree[x]
        coverDegree = cover.sum()

        self.coverList2[x] = cover[cover == True].index
        return coverDegree

    def updateCoverDegree(self, x, A, AA):
        self.coverList[x] = self.coverList[x] ^ (self.coverList[x] & A)
        self.coverList2[x] = self.coverList2[x] ^ (self.coverList2[x] & AA)
        return len(self.coverList[x]) + len(self.coverList2[x])

    def setXXX(self, x):
        y = self.y_train.loc[x]
        self.xxx = self.X_train.loc[~(self.y_train == y)].index

    def save2json(self, path=None):
        index_s = self.minRules.keys()
        X_sub = self.X_train.loc[index_s]
        y_sub = self.y_train.loc[index_s]
        data = {'rules': self.minRules, 'data': X_sub.values.tolist(),
                'label': y_sub.values.tolist(), 'consDegree': pd.Series(self.consDegree).loc[index_s].to_dict()}
        if path is None:
            path = f'result/rule/{self.name}/{self.alpha}.json'
        dirname, basename = os.path.split(path)
        if os.path.exists(dirname) == False and len(dirname.strip()) > 0:
            os.makedirs(dirname)
        if self.verbose > 0:
            print(f'规则集结果保存到{path}')
        with open(path, 'w') as fw:
            json.dump(data, fw)

    def calReduct(self, x):
        currConsDegree = 0
        red = []
        initConsDegree = self.consDegree[x]

        while initConsDegree > currConsDegree:
            currConsDegree, attr_idx = self.calConsDegree()
            red.append(int(attr_idx))
        return red


    def score(self,X, y):
        current_decision = self.predict(X)
        accuracy = sum(current_decision == y) / len(y)
        return accuracy

    def predict(self, X):
        rule_num = len(self.minRules)
        row = len(X)
        deci_rule_set = self.y_train.loc[self.minRules.keys()].values
        cover_object = np.zeros((row, rule_num))

        for i, (k, idx) in enumerate(self.minRules.items()):
            consistence = np.array([self.consDegree[k]] * row).reshape((-1, 1))
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
        # print(current_decision)
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

    def get_params(self, deep=False):
        return {'alpha': self.alpha, 'name': self.name, 'verbose': self.verbose, 'sample_percent': self.sample_percent,
                'early':self.early}

    def set_params(self, **params):
        if not params:
            return self
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def save2excel(self, path=None):
        index_s = self.minRules.keys()
        self.data2excel.append(len(index_s))

        df = pd.DataFrame([self.data2excel],
                          columns=["数据集", "训练样本数目", "特征数目", "标记数目", "模型阈值", "覆盖度阈值", "BatchSize比例", "开始时间", "结束时间", "总时间",
                                   "属性值约简数目","选择特征数目", "规则集大小"])
        if path is None:
            path = f"result/BSRC_{self.early}_{self.sample_percent}.csv"
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
    def calCD_0(self):
        dd = {}
        for x in self.rule_idx:
            coverDegree = self.calCoverDegree(x) + self.calCoverDegree2(x)
            if dd.get(coverDegree) is None:
                dd[coverDegree] = [x]
            else:
                dd[coverDegree].append(x)
        cd = max(dd.keys())
        kk = min(dd[cd])
        return kk, cd
    def reductionRule_0(self,idxList):
        # self.X_train_copy = self.X_train.copy()
        # self.y_train_copy = self.y_train.copy()
        self.X_train_copy.drop(idxList,inplace=True)
        self.y_train_copy.drop(idxList, inplace=True)
        idx = self.X_train_copy.index
        while True:
            kk,cd = self.calCD_0()
            if cd <= self.early:
                self.rule_idx=[]
                break
            self.minRules[kk]=self.allReduct[kk]
            self.rule_idx = self.rule_idx ^ self.coverList[kk]
            if self.verbose > 0:
                print(f'选择 {kk},覆盖度:{cd}')
            self.X_train_copy.drop(self.coverList2[kk], inplace=True)
            self.y_train_copy.drop(self.coverList2[kk], inplace=True)

            self.minRules[kk] = self.allReduct[kk]
            if self.rule_idx.size == 0:
                self.rule_idx = []
                if self.verbose > 0:
                    print('规则集pool为空!')
                break
    def reductionRule(self, idxList):
        self.y_train_copy.drop(idxList, inplace=True)
        kk, cd = self.calCD()
        while True:
            if cd <= self.early:
                self.rule_idx = []
                if self.verbose > 0:
                    print(f'覆盖度小于阈值{self.early}, early stop')
                break
            if self.verbose > 0:
                print(f'选择 {kk},覆盖度:{cd}')
            self.minRules[kk] = self.allReduct[kk]
            #   更新规则池
            self.rule_idx = self.rule_idx ^ self.coverList[kk]
            #   删除无关样本
            self.y_train_copy.drop(self.coverList2[kk], inplace=True)
            if self.rule_idx.size == 0:
                self.rule_idx = []
                if self.verbose > 0:
                    print('规则集pool为空!')
                break
            kk, cd = self.updateCD(kk)

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

    def calReductAll(self):
        self.data2excel.append(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        start = time.time()
        total = len(self.idx)
        S = 0
        while True:
            b = max(math.ceil(total * self.sample_percent),1)
            idxList = []
            for x in self.idx:
                S += 1
                idxList.append(x)
                self.setXXX(x)
                if self.mode==1:
                    self.dis_a = abs(self.X_train.loc[x] - self.X_train.loc[self.xxx])
                    cons = min(self.dis_a.max(axis=1).min() + self.alpha, 1)
                    if cons == 0:
                        raise ValueError(f'{x}的下近似为0,即存在不一致的样本')
                    self.consDegree[x] = cons
                    rule = self.calReduct(x)
                elif self.mode==2:
                    cons = self.calConsistenceDegree(x)
                    self.consDegree[x] = cons
                    rule = self.calReduct_0(x)
                else:
                    raise ValueError('mode 不存在！')
                self.allReduct[x] = rule
                self.dis_a = None
                self.dis_B = None
                self.setXXX(x)
                self.removeRedundancy(x)
                self.xxx = None
                b -= 1
                if b == 0:
                    break
            self.rule_idx.extend(idxList)
            self.rule_idx = pd.Index(self.rule_idx)
            if self.mode==1:
                self.reductionRule(idxList)
            elif self.mode==2:
                self.reductionRule_0(idxList)
            self.idx = self.y_train_copy.index
            if len(self.idx) == 0:
                self.rule_idx = pd.Index(self.rule_idx)
                break
        end = time.time()
        self.data2excel.append(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        self.data2excel.append(end - start)
        print(f'运行时间:{self.data2excel[-1]}')
        self.data2excel.append(S)

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.y_train_copy = copy.deepcopy(self.y_train)
        self.X_train_copy = copy.deepcopy(self.X_train)
        self.idx = self.y_train_copy.index
        self.data2excel = [self.name, len(self.idx), self.X_train.shape[1], len(self.y_train_copy.unique()),
                           self.alpha, self.early, self.sample_percent]
        self.calReductAll()
        a = []
        for key, value in self.minRules.items():
            for i in value:
                if i not in a:
                    a.append(i)
        print(f'选择特征数目={len(a)}')
        self.data2excel.append(len(a))
        self.save2excel()
        self.save2json()

if __name__ == '__main__':
    # name = 'sonar'#Glass,spect,Iono

    name='test'#['wine','Sonar','Glass','spect','Iono',"Libras","WDBC","QSAR","Segm","Wave","Cont","Spam","Park"]:
    dataPath = f'../standard/{name}.csv'
    from paper.common import readDataFile
    from sklearn.model_selection import train_test_split

    X, y = readDataFile(dataPath)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    obj = BatchSampleRuleClassifier(alpha=0, name=name, early=0, sample_percent=0.001, verbose=0)
    obj.fit(X_train, y_train)
    print(f"{name}准确率：{obj.score(X_test, y_test)}")

