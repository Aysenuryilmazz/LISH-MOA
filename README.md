## LISH-MOA Challenge  
<img src="https://github.com/mustafahakkoz/LISH-MOA/blob/hakkoz/images/etki.jpg" height="150" /> 

Final Project for 2020-2021 Spring CSE4088 - Introduction to Machine Learning, CSE, Marmara University.

- **Ayşenur YILMAZ** [@Aysenuryilmazz](https://github.com/Aysenuryilmazz)
- **Mustafa Abdullah HAKKOZ** [@mustafahakkoz](https://github.com/mustafahakkoz)

In this project, we aimed to work on an kaggle challenge [Mechanisms of Action (MoA) Prediction](https://www.kaggle.com/c/lish-moa). We couldn't be able to submit our solution in time, but dataset was pretty interesting to work with so we have completed the project anyway.

---

### Dataset Specifications and Challenges
- Multilabel problem with 207 binary target attributes (possible MoA patterns)
- Huge number of dimensions (mostly gene and cell expression values) in contrast to training set size (23814 rows × 876 columns)
- The same goes for test set (3982 rows × 876 columns) 
- We had to predict whether a particular gene expression&cell viability sample corresponds to a specific MoA. One single drug sample can have multiple MoAs (out of 207 different MoAs).

---
### Implementation Steps
We tried out 2 algorithms **XGBoost** and a **Neural Network (Keras)**. The main ideas for the implementation are:

- Transforming all columns to normal distribution with help of RankGauss (**QuantileTransformer**).  
<img src="https://github.com/mustafahakkoz/LISH-MOA/blob/hakkoz/images/resim1.png" height="150" /> <img src="https://github.com/mustafahakkoz/LISH-MOA/blob/hakkoz/images/resim2.png" height="150" /> 

- Adding **PCA** dimensions to dataset.  
**original dataset** has 772 gene expressions, 100 cell expressions  
**PCA outputs** 463 gene expressions, 60 cell expressions  
**new dimensions** are 4 + (772 + 100) + (463 + 60) = 1399
- Feature elimination with **variance threshold** (0.9), new dimensions are 1014.
- Applying **CountEncoder** to only categorical variables "cp_dose" and "cp_time".
- Scikit-learn provides several cross validators with stratification. 
However, these cross validators do not offer the ability to stratify multilabel data. [This iterative-stratification project](https://github.com/trent-b/iterative-stratification) offers implementations of **MultilabelStratifiedKFold** and we used it in our project.
- **CV predictions**. Generally, out-of-fold predictions improves scores so that we used 5 fold to make predictions on test dataset and take average on them. This technique resembles bagging ensemble.
- **Multilabel logloss** for the evaluation metric. M is the number of MoA and N is the number of samples.  
<img src="https://github.com/mustafahakkoz/LISH-MOA/blob/hakkoz/images/resim3.png" height="150" />
- Using **MultiOutputClassifier** of [Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) together with XGBoost (that runs the same model for every target in a multilabel problem) we achieved **0.024239** score on the test set.
- Deep Learning model had better scores: **0.0159892**. It uses same preprocessing steps with XGBoost and a **StandardScaler** on top of it.
- It has 2 hidden layers with **WeightNormalization** along with **BatchNormalization** and **Dropout** layers (with rates of 0.2, 0.5 and 0.5).  
<img src="https://github.com/mustafahakkoz/LISH-MOA/blob/hakkoz/images/resim4.png" weight="150" />
- It also uses **ReduceLearningRateOnPlateau** with factor of 0.1 and patience of 3.
- And finally the model is trained by **LookAhead** with **Adam** optimizer with sync period of 10, epoch number of 35 and batch size of 128. 

More information can be found in [project report](https://github.com/mustafahakkoz/LISH-MOA/blob/hakkoz/CSE4088_Aysenur_Yilmaz_Mustafa_Hakkoz_FinalReport.pdf) and [presentation slides](https://github.com/mustafahakkoz/LISH-MOA/blob/hakkoz/CSE4088_Aysenur_Yilmaz_Mustafa_Hakkoz_Presentation.ppt).

---
### Repo Content
- **Exploratory Data Analysis:** [ml-project-lish-moa-eda.ipynb](https://github.com/mustafahakkoz/LISH-MOA/blob/hakkoz/ml-project-lish-moa-eda.ipynb)  
- **XGBoost notebook:** [ml-project-lish-moa-xgboost.ipynb](https://github.com/mustafahakkoz/LISH-MOA/blob/hakkoz/ml-project-lish-moa-xgboost.ipynb)   
- **Neural Network notebook:** [ml-project-lish-moa-nn.ipynb](https://github.com/mustafahakkoz/LISH-MOA/blob/hakkoz/ml-project-lish-moa-nn.ipynb)   
- **Final Project Report:** [CSE4088_Aysenur_Yilmaz_Mustafa_Hakkoz_FinalReport.pdf](https://github.com/mustafahakkoz/LISH-MOA/blob/hakkoz/CSE4088_Aysenur_Yilmaz_Mustafa_Hakkoz_FinalReport.pdf)   
- **Project Presentation:** [CSE4088_Aysenur_Yilmaz_Mustafa_Hakkoz_Presentation.ppt](https://github.com/mustafahakkoz/LISH-MOA/blob/hakkoz/CSE4088_Aysenur_Yilmaz_Mustafa_Hakkoz_Presentation.ppt)   
---
### Online Notebooks

- [**EDA**](https://www.kaggle.com/hakkoz/ml-project-lish-moa-eda)
- [**Classification with XGBOOST**](https://www.kaggle.com/hakkoz/ml-project-lish-moa-xgboost)
- [**Classification with NN**](https://www.kaggle.com/hakkoz/ml-project-lish-moa-nn)

---
### Additional Info
The Connectivity Map, a project within the Broad Institute of MIT and Harvard, the Laboratory for Innovation Science at Harvard (LISH), and the NIH Common Funds Library of Integrated Network-Based Cellular Signatures (LINCS), present this challenge with the goal of advancing drug development through improvements to MoA prediction algorithms.

> What is the Mechanism of Action (MoA) of a drug? And why is it important?

>> In the past, scientists derived drugs from natural products or were inspired by traditional remedies. Very common drugs, such as paracetamol, known in the US as acetaminophen, were put into clinical use decades before the biological mechanisms driving their pharmacological activities were understood. Today, with the advent of more powerful technologies, drug discovery has changed from the serendipitous approaches of the past to a more targeted model based on an understanding of the underlying biological mechanism of a disease. In this new framework, scientists seek to identify a protein target associated with a disease and develop a molecule that can modulate that protein target. As a shorthand to describe the biological activity of a given molecule, scientists assign a label referred to as mechanism-of-action or MoA for short.


> How do we determine the MoAs of a new drug?

>> One approach is to treat a sample of human cells with the drug and then analyze the cellular responses with algorithms that search for similarity to known patterns in large genomic databases, such as libraries of gene expression or cell viability patterns of drugs with known MoAs.

>> In this competition, you will have access to a unique dataset that combines gene expression and cell viability data. The data is based on a new technology that measures simultaneously (within the same samples) human cells’ responses to drugs in a pool of 100 different cell types (thus solving the problem of identifying ex-ante, which cell types are better suited for a given drug). In addition, you will have access to MoA annotations for more than 5,000 drugs in this dataset.

>> As is customary, the dataset has been split into testing and training subsets. Hence, your task is to use the training dataset to develop an algorithm that automatically labels each case in the test set as one or more MoA classes. Note that since drugs can have multiple MoA annotations, the task is formally a multi-label classification problem.

> How to evaluate the accuracy of a solution?

>> Based on the MoA annotations, the accuracy of solutions will be evaluated on the average value of the logarithmic loss function applied to each drug-MoA annotation pair.

>>If successful, you’ll help to develop an algorithm to predict a compound’s MoA given its cellular signature, thus helping scientists advance the drug discovery process.
