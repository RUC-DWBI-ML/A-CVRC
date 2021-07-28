#  A-CVRC（An Accelerator for Rule Induction in Fuzzy Rough Theory）


### What is it?

Rule-based classifier, that extract a subset of induced rules to efficiently learn/mine while preserving the discernibility information, plays an essential role in human-explainable artificial intelligence. However, in this era of big data, rule induction on the whole datasets is computationally intensive. Till now, to our best knowledge, there is no method which focus on expediting rule induction. It is the first time this study considers the acceleration technique to reduce the scale of computation in rule induction. We propose an accelerator for rule induction based the fuzzy rough-set theory, which can avoid redundant computation and accelerate the building of rule classifier.

### What is this package?

### Dependencies and Installation

  - python 3.6
  - numpy 1.18.1
  - pandas 1.0.1
  - sklearn 0.20.1
  
Before running the datasets, they need to be normalized. So we can use the "origin2standard" from "common" class to uniform the datasets which mainly use the MinMaxScaler method to deal with datasets. 


## run文件是需要运行的一些代码，调整需要跑的数据集以及算法对应参数

- ``````
  ### fun1用运行RuleClassifier分类器，输出RuleClassifier的准确率
  def fun1():
  
  ### fun2用来运行BatchSampleRuleClassifier分类器，输出BatchSampleRuleClassifier的准确率
  def fun2():
  
  ### fun3用来运行LEM2RuleClassifier分类器，输出LEM2RuleClassifier的准确率
  def fun3():
  
  ### fun4用来运行VCDOLEMRuleClassifier分类器，输出VCDOLEMRuleClassifier的准确率
  def fun4():
  
  ### fun5用来运行FVPRS分类器，输出FVPRS的准确率
  def fun5():
  ``````
  
- ``````
  单个分类器：
  #	RuleClassifier普通算法
  RuleClassifier(mode=2)
  #	RuleClassifier加速算法
  RuleClassifier(mode=1)
  #	基于Batch的算法，以它为基准做T-test
  BatchSampleRuleClassifier()
  #	LEM2Classifier普通算法
  LEM2Classifier(mode=2)
  #	LEM2Classifier加速算法
  LEM2Classifier(mode=1)
  #	VCDOLEMClassifier普通算法
  VCDOLEMClassifier(mode=2)
  #	VCDOLEMClassifier加速算法
  VCDOLEMClassifier(mode=1)
  #	FVPRS
  FVPRS()
  
  ``````
	
	### Any issues
	
	If you have any good ideas or demands, please open an issue/discussion to let me know.
	If you have datasets that A-CVRC could not running, please also open an issue/discussion. I will record it (but I cannot guarantee to resolve it😛). 
