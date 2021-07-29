
from paper.common import  getCVAccuracy_noise
from paper.RuleClassifier import RuleClassifier
from paper.BatchSampleRuleClassifier import BatchSampleRuleClassifier


### Datasets:
### ['Iono','Libras','QSAR','Cont','Segm','Spam',"Wave",'texture','optdigits','Park', 'sat','Musk2',"marketing",'thyroid','ring','coil2000',
###              'crowd','pendigits','nursery','eeg',"Gamma","Letter","Krkopt","Credit","adult",'Shuttle','Sensorless',]

### fun1: running the RuleClassifier and print the accuracy of A-CVRC
def fun1():
    #   "mode=1" and "mode=2" indicates the acclerator algorithm and non-acclerate version, respectively
    dataList = ['test']
    clf = RuleClassifier()
    getCVAccuracy_noise(dataList, 'RuleClassifier', clf, name=True, noise=0, alpha=0,
                          verbose=0, mode=1,save = True)

### fun2: Running the BatchSampleRuleClassifier and print the accuracy of A-BSRC
def fun2():
    dataList = ['test']
    ### If the number of samples is greater than 20000, the value of sample_percent is set to 0.001
    sample_percent = 0.01
    clf = BatchSampleRuleClassifier()
    getCVAccuracy_noise(dataList, 'BatchSampleRuleClassifier', clf, name=True, noise=0,alpha=0, sample_percent=sample_percent, early=0, verbose=0)



if __name__ == '__main__':

    # fun1()
    # fun2()
