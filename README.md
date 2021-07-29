#  A-CVRC（An Accelerator for Rule Induction in Fuzzy Rough Theory）


### What is it?

Rule-based classifier, that extract a subset of induced rules to efficiently learn/mine while preserving the discernibility information, plays an essential role in human-explainable artificial intelligence. However, in this era of big data, rule induction on the whole datasets is computationally intensive. Till now, to our best knowledge, there is no method which focus on expediting rule induction. It is the first time this study considers the acceleration technique to reduce the scale of computation in rule induction. We propose an accelerator for rule induction based the fuzzy rough-set theory, which can avoid redundant computation and accelerate the building of rule classifier.

### Requirements

  - python 3.6
  - numpy 1.18.1
  - pandas 1.0.1
  - sklearn 0.20.1

### Dataset

We conducted numerical experiments on a series of UCI and KEEL datasets. Before running the datasets, they need to be normalized. So we can use the "origin2standard" from "common" class to uniform the datasets which mainly use the MinMaxScaler method to deal with datasets. 

### Usage
Running the ``run.py`` file

### Any issues

If you have any good ideas or demands, please open an issue/discussion to let me know.
If you have datasets that A-CVRC could not running, please also open an issue/discussion. I will record it (but I cannot guarantee to resolve it😛). 
