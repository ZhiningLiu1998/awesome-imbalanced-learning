<!-- <h1 align="center"> Awesome Imbalanced Learning </h1> -->

![](https://raw.githubusercontent.com/ZhiningLiu1998/figures/master/awesome-imbalanced-learning/awesome_imbalanced_learning_header.png)

<p align="center">
  <img src="https://awesome.re/badge.svg">
  <a href="https://github.com/ZhiningLiu1998/awesome-imbalanced-learning">
    <img src="https://img.shields.io/badge/ZhiningLiu1998-AwesomeIL-orange">
  </a>
  <img src="https://img.shields.io/github/stars/ZhiningLiu1998/awesome-imbalanced-learning">
  <img src="https://img.shields.io/github/forks/ZhiningLiu1998/awesome-imbalanced-learning">
  <img src="https://img.shields.io/github/issues/ZhiningLiu1998/awesome-imbalanced-learning">
  <img src="https://img.shields.io/github/license/ZhiningLiu1998/awesome-imbalanced-learning">
</p>

**A curated list of awesome imbalanced learning papers, codes, frameworks and libraries.**

**Class-imbalance** (also known as the long-tail problem) is the fact that the classes are not represented equally in a classification problem, which is quite common in practice. For instance, fraud detection, prediction of rare adverse drug reactions and prediction gene families. Failure to account for the class imbalance often causes inaccurate and decreased predictive performance of many classification algorithms. Imbalanced learning aims to tackle the class imbalance problem to learn an unbiased model from imbalanced data.

Inspired by [awesome-machine-learning](https://github.com/josephmisiti/awesome-machine-learning). **Contributions are welcomed!**

- Frameworks and libraries are grouped by *programming language*.
- Research papers are grouped by *research field*.
- There are numerous papers in this field of research, so this list is not intended to be exhaustive. We aim to keep only the "awesome" works that either *have a good impact* or have been *published in reputed top conferences/journals*.


<h3>
<font color='red'>What's new: </font>
</h3>

- [10/2021] Add papers from ICCV.
- [08/2021] Add papers on cost-sensitive ensemble learning.
- [05/2021] Add a scikit-learn-like package [imbalanced-ensemble](https://github.com/ZhiningLiu1998/imbalanced-ensemble) [[Github](https://github.com/ZhiningLiu1998/imbalanced-ensemble)][[Documentation](https://imbalanced-ensemble.readthedocs.io/)].
- [05/2021] Add papers from ICLR/CVPR/WSDM/TKDE/TNNLS.
- <details>
  <summary>More</summary>
      <LI> [11/2020] Add papers from NeurIPS/ICLR. <br>
  </details>    

**Disclosure:** Zhining Liu is an author on the following works: [imbalanced-ensemble](https://github.com/ZhiningLiu1998/imbalanced-ensemble), [Self-paced Ensemble](https://github.com/ZhiningLiu1998/self-paced-ensemble), [MESA](https://github.com/ZhiningLiu1998/mesa).

# Table of Contents

- [Table of Contents](#table-of-contents)
- [Frameworks and Libraries](#frameworks-and-libraries)
    - [Python](#python)
    - [R](#r)
    - [Java](#java)
    - [Scalar](#scalar)
    - [Julia](#julia)
- [Research Papers](#research-papers)
  - [Surveys](#surveys)
  - [Ensemble Learning](#ensemble-learning)
      - [*General ensemble*](#general-ensemble)
      - [*Boosting-based*](#boosting-based)
      - [*Bagging-based*](#bagging-based)
      - [*Cost-sensitive ensemble*](#cost-sensitive-ensemble)
  - [Data resampling](#data-resampling)
      - [*Over-sampling*](#over-sampling)
      - [*Under-sampling*](#under-sampling)
      - [*Hybrid-sampling*](#hybrid-sampling)
  - [Cost-sensitive Learning](#cost-sensitive-learning)
  - [Deep Learning](#deep-learning)
      - [*Surveys*](#surveys-1)
      - [*Hard example mining*](#hard-example-mining)
      - [*Loss function engineering*](#loss-function-engineering)
      - [*Meta-learning*](#meta-learning)
      - [*Representation Learning*](#representation-learning)
      - [*Posterior Recalibration*](#posterior-recalibration)
      - [*Semi/Self-supervised Learning*](#semiself-supervised-learning)
      - [*Curriculum Learning*](#curriculum-learning)
      - [*Two-phase Training*](#two-phase-training)
      - [*Network Architecture*](#network-architecture)
      - [*Graph Neural Networks*](#graph-neural-networks)
      - [*Deep Generative Model*](#deep-generative-model)
      - [*Imbalanced Regression*](#imbalanced-regression)
  - [Anomaly Detection](#anomaly-detection)
- [Miscellaneous](#miscellaneous)
  - [Datasets](#datasets)
  - [Github Repositories](#github-repositories)
    - [*Algorithms & Utilities & Jupyter Notebooks*](#algorithms--utilities--jupyter-notebooks)
    - [*Paper list*](#paper-list)
    - [*Slides*](#slides)


# Frameworks and Libraries

### Python
- [**imbalanced-ensemble**](https://imbalanced-ensemble.readthedocs.io/) [[**Github**](https://github.com/ZhiningLiu1998/imbalanced-ensemble)][[**Documentation**](https://imbalanced-ensemble.readthedocs.io/)][[**Gallery**](https://imbalanced-ensemble.readthedocs.io/en/latest/auto_examples/index.html#)] 
  > **NOTE:** written in python, easy to use.
  - `imbalanced-ensemble` is a Python toolbox for quick implementing and deploying ***ensemble learning algorithms*** on class-imbalanced data. It is featured for: 
    - (i) Unified, easy-to-use APIs, detailed [documentation](https://imbalanced-ensemble.readthedocs.io/) and [examples](https://imbalanced-ensemble.readthedocs.io/en/latest/auto_examples/index.html#). 
    - (ii) Capable for multi-class imbalanced learning out-of-box. 
    - (iii) Optimized performance with parallelization when possible using [joblib](https://github.com/joblib/joblib).
    - (iv) Powerful, customizable, interactive training logging and visualizer.
    - (v) Full compatibility with other popular packages like [scikit-learn](https://scikit-learn.org/stable/) and [imbalanced-learn](https://imbalanced-learn.org/stable/).
  - Currently (v0.1.4), it includes more than 15 ensemble algorithms based on ***re-sampling*** and ***cost-sensitive learning*** (e.g., *SMOTEBoost/Bagging, RUSBoost/Bagging, AdaCost, EasyEnsemble, BalanceCascade, SelfPacedEnsemble*, ...).

- [**imbalanced-learn**](https://imbalanced-learn.org/stable/) [[**Github**](https://github.com/scikit-learn-contrib/imbalanced-learn)][[**Documentation**](https://imbalanced-learn.org/stable/)][[**Paper**](http://10.187.70.34/www.jmlr.org/papers/volume18/16-365/16-365.pdf)] 
  > **NOTE:** written in python, easy to use.
  - `imbalanced-learn` is a python package offering a number of ***re-sampling*** techniques commonly used in datasets showing strong between-class imbalance. It is compatible with [scikit-learn](https://scikit-learn.org/stable/) and is part of [scikit-learn-contrib](https://github.com/scikit-learn-contrib) projects. 
  - Currently (v0.8.0), it includes 21 different re-sampling techniques, including over-sampling, under-sampling and hybrid ones (e.g., *SMOTE, ADASYN, TomekLinks, NearMiss, OneSideSelection*, SMOTETomek, ...) 
  - This package also provides many utilities, e.g., *Batch generator for Keras/TensorFlow*, see [API reference](https://imbalanced-learn.org/stable/references/index.html#api).

- [**smote_variants**](https://smote-variants.readthedocs.io/en/latest/) [[**Documentation**](https://smote-variants.readthedocs.io/en/latest/)][[**Github**](https://github.com/analyticalmindsltd/smote_variants)] - A collection of 85 minority ***over-sampling*** techniques for imbalanced learning with multi-class oversampling and model selection features (All writen in Python, also support R and Julia).

### R
- [**smote_variants**](https://smote-variants.readthedocs.io/en/latest/) [[**Documentation**](https://smote-variants.readthedocs.io/en/latest/)][[**Github**](https://github.com/analyticalmindsltd/smote_variants)] - A collection of 85 minority ***over-sampling*** techniques for imbalanced learning with multi-class oversampling and model selection features (All writen in Python, also support R and Julia).
- [**caret**](https://cran.r-project.org/web/packages/caret/index.html) [[**Documentation**](http://topepo.github.io/caret/index.html)][[**Github**](https://github.com/topepo/caret)] - Contains the implementation of Random under/over-sampling.
- [**ROSE**](https://cran.r-project.org/web/packages/ROSE/index.html) [[**Documentation**](https://www.rdocumentation.org/packages/ROSE/versions/0.0-3)] - Contains the implementation of [ROSE](https://journal.r-project.org/archive/2014-1/menardi-lunardon-torelli.pdf) (Random Over-Sampling Examples).
- [**DMwR**](https://cran.r-project.org/web/packages/DMwR/index.html) [[**Documentation**](https://www.rdocumentation.org/packages/DMwR/versions/0.4.1)] - Contains the implementation of [SMOTE](https://arxiv.org/pdf/1106.1813.pdf) (Synthetic Minority Over-sampling TEchnique).

### Java
- [**KEEL**](https://sci2s.ugr.es/keel/description.php) [[**Github**](https://github.com/SCI2SUGR/KEEL)][[**Paper**](https://sci2s.ugr.es/sites/default/files/ficherosPublicaciones/0758_Alcalaetal-SoftComputing-Keel1.0.pdf)] - KEEL provides a simple ***GUI based*** on data flow to design experiments with different datasets and computational intelligence algorithms (***paying special attention to evolutionary algorithms***) in order to assess the behavior of the algorithms. This tool includes many widely used imbalanced learning techniques such as (evolutionary) over/under-resampling, cost-sensitive learning, algorithm modification, and ensemble learning methods. 
    
    > **NOTE:** wide variety of classical classification, regression, preprocessing algorithms included.

### Scalar
- [**undersampling**](https://github.com/NestorRV/undersampling) [[**Documentation**](https://nestorrv.github.io/)][[**Github**](https://github.com/NestorRV/undersampling)] - A Scala library for ***under-sampling and their ensemble variants*** in imbalanced classification.

### Julia
- [**smote_variants**](https://smote-variants.readthedocs.io/en/latest/) [[**Documentation**](https://smote-variants.readthedocs.io/en/latest/)][[**Github**](https://github.com/analyticalmindsltd/smote_variants)] - A collection of 85 minority ***over-sampling*** techniques for imbalanced learning with multi-class oversampling and model selection features (All writen in Python, also support R and Julia).

# Research Papers

## Surveys

- **Learning from imbalanced data (IEEE TKDE, 2009, 6000+ citations) [[**Paper**](https://www.sci-hub.shop/10.1109/tkde.2008.239)]**
  - Highly cited, classic survey paper. It systematically reviewed the popular solutions, evaluation metrics, and challenging problems in future research in this area (as of 2009). 

- **Learning from imbalanced data: open challenges and future directions (2016, 900+ citations) [[**Paper**](https://www.researchgate.net/publication/301596547_Learning_from_imbalanced_data_Open_challenges_and_future_directions)]**
  - This paper concentrates on the open issues and challenges in imbalanced learning, i.e., extreme class imbalance, imbalance in online/stream learning, multi-class imbalanced learning, and semi/un-supervised imbalanced learning.

- **Learning from class-imbalanced data: Review of methods and applications (2017, 900+ citations) [[**Paper**](https://www.researchgate.net/publication/311977198_Learning_from_class-imbalanced_data_Review_of_methods_and_applications)]**
  - A recent exhaustive survey of imbalanced learning methods and applications, a total of 527 papers were included in this study. It provides several detailed taxonomies of existing methods and also the recent trend of this research area.

## Ensemble Learning

#### *General ensemble*
<!-- - **General ensemble** -->
  
  - **Self-paced Ensemble (ICDE 2020, 20+ citations) [[**Paper**](https://arxiv.org/pdf/1909.03500v3.pdf)][[**Code**](https://github.com/ZhiningLiu1998/self-paced-ensemble)][[**Slides**](https://zhiningliu.com/files/ICDE_2020_self_paced_ensemble_slides.pdf)][[**Zhihu/知乎**](https://zhuanlan.zhihu.com/p/86891438)][[**PyPI**](https://pypi.org/project/self-paced-ensemble/)]**
    > **NOTE:** versatile solution with outstanding performance and computational efficiency.

  - **MESA: Boost Ensemble Imbalanced Learning with MEta-SAmpler (NeurIPS 2020) [[**Paper**](https://arxiv.org/pdf/2010.08830.pdf)][[**Code**](https://github.com/ZhiningLiu1998/mesa)][[**Video**](https://studio.slideslive.com/web_recorder/share/20201020T134559Z__NeurIPS_posters__17343__mesa-effective-ensemble-imbal?s=d3745afc-cfcf-4d60-9f34-63d3d811b55f)][[**Zhihu/知乎**](https://zhuanlan.zhihu.com/p/268539195)]**
    > **NOTE:** learning an optimal sampling policy directly from data.

  - **Exploratory Undersampling for Class-Imbalance Learning (IEEE Trans. on SMC, 2008, 1300+ citations) [[**Paper**](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/2009-IEEE%20TSMCpartB%20Exploratory%20Undersampling%20for%20Class%20Imbalance%20Learning.pdf)]**
    > **NOTE:** simple but effective solution.
    - EasyEnsemble [[**Code**](https://github.com/ZhiningLiu1998/imbalanced-ensemble/blob/main/imbalanced_ensemble/ensemble/under_sampling/easy_ensemble.py)] 
    - BalanceCascade [[**Code**](https://github.com/ZhiningLiu1998/imbalanced-ensemble/blob/main/imbalanced_ensemble/ensemble/under_sampling/balance_cascade.py)]
  

#### *Boosting-based*
<!-- - **Boosting-based** -->

  - **AdaBoost (1995, 18700+ citations) [[**Paper**](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/1997-JCSS-Schapire-A%20Decision-Theoretic%20Generalization%20of%20On-Line%20Learning%20(AdaBoost).pdf)][[**Code**](https://github.com/scikit-learn/scikit-learn/blob/95d4f0841/sklearn/ensemble/_weight_boosting.py#L285)]** - Adaptive Boosting with C4.5
  
  - **DataBoost (2004, 570+ citations) [[**Paper**](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/2004-SIGKDD-GuoViktor.pdf)]** - Boosting with Data Generation for Imbalanced Data
  
  - **SMOTEBoost (2003, 1100+ citations) [[**Paper**]](https://sci2s.ugr.es/keel/pdf/algorithm/congreso/2003-PKDD-SMOTEBoost-ChawlaLazarevicHallBowyer.pdf)[[**Code**](https://github.com/ZhiningLiu1998/imbalanced-ensemble/blob/main/imbalanced_ensemble/ensemble/over_sampling/smote_bagging.py)]** - Synthetic Minority Over-sampling TEchnique Boosting

  - **MSMOTEBoost (2011, 1300+ citations) [[**Paper**](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/2011-IEEE%20TSMC%20partC-%20GalarFdezBarrenecheaBustinceHerrera.pdf)]** - Modified Synthetic Minority Over-sampling TEchnique Boosting
  
  - **RAMOBoost (2010, 140+ citations) [[**Paper**](https://www.ele.uri.edu/faculty/he/PDFfiles/ramoboost.pdf)] [[**Code**](https://github.com/dialnd/imbalanced-algorithms/blob/master/ramo.py#L133)]** - Ranked Minority Over-sampling in Boosting
  
  - **RUSBoost (2009, 850+ citations) [[**Paper**](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/2010-IEEE%20TSMCpartA-RUSBoost%20A%20Hybrid%20Approach%20to%20Alleviating%20Class%20Imbalance.pdf)] [[**Code**](https://github.com/ZhiningLiu1998/imbalanced-ensemble/blob/main/imbalanced_ensemble/ensemble/under_sampling/rus_boost.py)]** - Random Under-Sampling Boosting

  - **AdaBoostNC (2012, 350+ citations) [[**Paper**](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/2012-wang-IEEE_SMC_B.pdf)]** - Adaptive Boosting with Negative Correlation Learning
  
  - **EUSBoost (2013, 210+ citations) [[**Paper**](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/2013-galar-PR.pdf)]** - Evolutionary Under-sampling in Boosting

#### *Bagging-based*
<!-- - **Bagging-based** -->

  - **Bagging (1996, 20000+ citations) [[**Paper**](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/1996-ML-Breiman-Bagging%20Predictors.pdf)][[**Code**](https://github.com/scikit-learn/scikit-learn/blob/95d4f0841/sklearn/ensemble/_bagging.py#L433)]** - Bagging predictor

  - **Diversity Analysis on Imbalanced Data Sets by Using Ensemble Models (2009, 400+ citations) [[**Paper**](https://sci2s.ugr.es/keel/pdf/algorithm/congreso/2009-IEEE%20CIDM-WangYao.pdf)]**
    - **UnderBagging** [[**Code**](https://github.com/ZhiningLiu1998/imbalanced-ensemble/blob/main/imbalanced_ensemble/ensemble/under_sampling/under_bagging.py)] 
    - **OverBagging** [[**Code**](https://github.com/ZhiningLiu1998/imbalanced-ensemble/blob/main/imbalanced_ensemble/ensemble/over_sampling/over_bagging.py)] 
    - **SMOTEBagging** [[**Code**](https://github.com/ZhiningLiu1998/imbalanced-ensemble/blob/main/imbalanced_ensemble/ensemble/over_sampling/smote_bagging.py)]
  
#### *Cost-sensitive ensemble*
<!-- - **Cost-sensitive ensemble** -->

  - **AdaCost (ICML 1999, 800+ citations) [[**Paper**](https://www.researchgate.net/profile/Salvatore-Stolfo/publication/2628569_AdaCost_Misclassification_Cost-sensitive_Boosting/links/0fcfd50ca581d7016f000000/AdaCost-Misclassification-Cost-sensitive-Boosting.pdf)][[**Code**](https://github.com/ZhiningLiu1998/imbalanced-ensemble/blob/main/imbalanced_ensemble/ensemble/reweighting/adacost.py)]** - Misclassification Cost-sensitive boosting
  
  - **AdaUBoost (NIPS 1999, 100+ citations) [[**Paper**](https://proceedings.neurips.cc/paper/1998/file/df12ecd077efc8c23881028604dbb8cc-Paper.pdf)][[**Code**](https://github.com/ZhiningLiu1998/imbalanced-ensemble/blob/main/imbalanced_ensemble/ensemble/reweighting/adauboost.py)]** - AdaBoost with Unequal loss functions
  
  - **AsymBoost (NIPS 2001, 700+ citations) [[**Paper**](https://www.researchgate.net/profile/Michael-Jones-66/publication/2539888_Fast_and_Robust_Classification_using_Asymmetric_AdaBoost_and_a_Detector_Cascade/links/540731780cf23d9765a83ec1/Fast-and-Robust-Classification-using-Asymmetric-AdaBoost-and-a-Detector-Cascade.pdf)][[**Code**](https://github.com/ZhiningLiu1998/imbalanced-ensemble/blob/main/imbalanced_ensemble/ensemble/reweighting/asymmetric_boost.py)]** - Asymmetric AdaBoost and detector cascade

## Data resampling

#### *Over-sampling*
<!-- - **Over-sampling** -->

  - **ROS [[**Code**](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/over_sampling/_random_over_sampler.py)]** - Random Over-sampling 
  
  - **SMOTE (2002, 9800+ citations) [[**Paper**](https://arxiv.org/pdf/1106.1813.pdf)][[**Code**](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/over_sampling/_smote.py#L36)]** - Synthetic Minority Over-sampling TEchnique 
    
  - **Borderline-SMOTE (2005, 1400+ citations) [[**Paper**](https://sci2s.ugr.es/keel/keel-dataset/pdfs/2005-Han-LNCS.pdf)][[**Code**](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/over_sampling/_smote.py#L220)]** - Borderline-Synthetic Minority Over-sampling TEchnique 
  
  - **ADASYN (2008, 1100+ citations) [[**Paper**](https://sci2s.ugr.es/keel/pdf/algorithm/congreso/2008-He-ieee.pdf)][[**Code**](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/over_sampling/_adasyn.py)]** - ADAptive SYNthetic Sampling
  
  - **SPIDER (2008, 150+ citations) [[**Paper**](https://sci2s.ugr.es/keel/pdf/algorithm/congreso/stefanowski_selective_2008.pdf)][[**Code(Java)**](https://github.com/SCI2SUGR/KEEL/blob/master/src/keel/Algorithms/ImbalancedClassification/Resampling/SPIDER/SPIDER.java#L57)]** - Selective Preprocessing of Imbalanced Data
  
  - **Safe-Level-SMOTE (2009, 370+ citations) [[**Paper**](http://10.187.70.30/150.214.190.154/keel/keel-dataset/pdfs/2009-Bunkhumpornpat-LNCS.pdf)][[**Code(Java)**](https://github.com/SCI2SUGR/KEEL/blob/master/src/keel/Algorithms/ImbalancedClassification/Resampling/Safe_Level_SMOTE/Safe_Level_SMOTE.java#L58)]** - Safe Level Synthetic Minority Over-sampling TEchnique
  
  - **SVM-SMOTE (2009, 120+ citations) [[**Paper**](http://10.187.70.39/ousar.lib.okayama-u.ac.jp/files/public/1/19617/20160528004522391723/IWCIA2009_A1005.pdf)][[**Code**](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/over_sampling/_smote.py#L417)]** - SMOTE based on Support Vectors of SVM 
  
  - **MDO (2015, 150+ citations) [[**Paper**](https://ieeexplore.ieee.org/abstract/document/7163639)][[**Code**](https://github.com/analyticalmindsltd/smote_variants/blob/dedbc3d00b266954fedac0ae87775e1643bc920a/smote_variants/_smote_variants.py#L14513)]** - Mahalanobis Distance-based Over-sampling for *Multi-Class* imbalanced problems.
  
  > **NOTE:** See more over-sampling methods at [**smote-variants**](https://github.com/analyticalmindsltd/smote_variants#references).

#### *Under-sampling*
<!-- - **Under-sampling** -->

  - **RUS [[**Code**](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/under_sampling/_prototype_selection/_random_under_sampler.py)]** - Random Under-sampling
  
  - **CNN (1968, 2100+ citations) [[**Paper**](https://pdfs.semanticscholar.org/7c37/71fd6829630cf450af853df728ecd8da4ab2.pdf?_ga=2.137274553.882046879.1583413150-1712662047.1583413150)][[**Code**](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/under_sampling/_prototype_selection/_condensed_nearest_neighbour.py)]** - Condensed Nearest Neighbor
  
  - **ENN (1972, 1500+ citations) [[**Paper**](https://sci2s.ugr.es/keel/dataset/includes/catImbFiles/1972-Wilson-IEEETSMC.pdf)] [[**Code**](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/under_sampling/_prototype_selection/_edited_nearest_neighbours.py)]** - Edited Condensed Nearest Neighbor
   
  - **TomekLink (1976, 870+ citations) [[**Paper**](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4309452)][[**Code**](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/under_sampling/_prototype_selection/_tomek_links.py)]** - Tomek's modification of Condensed Nearest Neighbor 
  
  - **NCR (2001, 500+ citations) [[**Paper**](https://sci2s.ugr.es/keel/pdf/algorithm/congreso/2001-Laurikkala-LNCS.pdf)][[**Code**](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/under_sampling/_prototype_selection/_neighbourhood_cleaning_rule.py)]** - Neighborhood Cleaning Rule
  
  - **NearMiss-1 & 2 & 3 (2003, 420+ citations) [[**Paper**](https://sci2s.ugr.es/keel/pdf/specific/congreso/jzhang.pdf)][[**Code**](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/under_sampling/_prototype_selection/_nearmiss.py)]** - Several kNN approaches to unbalanced data distributions.
  
  - **CNN with TomekLink (2004, 2000+ citations) [[**Paper**](https://storm.cis.fordham.edu/~gweiss/selected-papers/batista-study-balancing-training-data.pdf)][[**Code(Java)**](https://github.com/SCI2SUGR/KEEL/blob/master/src/keel/Algorithms/ImbalancedClassification/Resampling/CNN_TomekLinks/CNN_TomekLinks.java#L58)]** - Condensed Nearest Neighbor + TomekLink
  
  - **OSS (2007, 2100+ citations) [[**Paper**](https://sci2s.ugr.es/keel/pdf/specific/congreso/kubat97addressing.pdf)][[**Code**](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/under_sampling/_prototype_selection/_one_sided_selection.py)]** - One Side Selection 
  
  - **EUS (2009, 290+ citations) [[**Paper**](https://www.mitpressjournals.org/doi/pdfplus/10.1162/evco.2009.17.3.275)]** - Evolutionary Under-sampling
  
  - **IHT (2014, 130+ citations) [[**Paper**](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.649.8727&rep=rep1&type=pdf)][[**Code**](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/under_sampling/_prototype_selection/_instance_hardness_threshold.py)]** - Instance Hardness Threshold 

#### *Hybrid-sampling*
<!-- - **Hybrid-sampling** -->

  - **A Study of the Behavior of Several Methods for Balancing Training Data (2004, 2000+ citations) [[**Paper**](https://www.researchgate.net/profile/Ronaldo-Prati/publication/220520041_A_Study_of_the_Behavior_of_Several_Methods_for_Balancing_machine_Learning_Training_Data/links/0d22cd91c989507054a2cf3b/A-Study-of-the-Behavior-of-Several-Methods-for-Balancing-machine-Learning-Training-Data.pdf)]**
    > **NOTE:** extensive experimental evaluation involving 10 different over/under-sampling methods.
    - **SMOTE-Tomek [[**Code**](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/combine/_smote_tomek.py)]**
    - **SMOTE-ENN [[**Code**](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/combine/_smote_enn.py)]**

  - **SMOTE-RSB (2012, 210+ citations) [[**Paper**](https://sci2s.ugr.es/sites/default/files/ficherosPublicaciones/1434_2012-Ramentol-KAIS.pdf)][[**Code**](https://smote-variants.readthedocs.io/en/latest/_modules/smote_variants/_smote_variants.html#SMOTE_RSB)]** - Hybrid Preprocessing using SMOTE and Rough Sets Theory
  
  - **SMOTE-IPF (2015, 180+ citations) [[**Paper**](https://sci2s.ugr.es/sites/default/files/ficherosPublicaciones/1824_2015-INS-Saez.pdf)][[**Code**](https://smote-variants.readthedocs.io/en/latest/_modules/smote_variants/_smote_variants.html#SMOTE_IPF)]** - SMOTE with Iterative-Partitioning Filter

## Cost-sensitive Learning

- **CSC4.5 (2002, 420+ citations) [[**Paper**](https://www.sci-hub.shop/10.1109/tkde.2002.1000348)][[**Code(Java)**](https://github.com/SCI2SUGR/KEEL/blob/master/src/keel/Algorithms/ImbalancedClassification/CSMethods/C45CS/C45CS.java#L48)]** - An instance-weighting method to induce cost-sensitive trees

- **CSSVM (2008, 710+ citations) [[**Paper**](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/2009-Chawla-IEEE_TSMCB-svm-imbalance.pdf)][[**Code(Java)**](https://github.com/SCI2SUGR/KEEL/blob/master/src/keel/Algorithms/ImbalancedClassification/CSMethods/C_SVMCost/svmClassifierCost.java#L60)]** - Cost-sensitive SVMs for highly imbalanced classification

- **CSNN (2005, 950+ citations) [[**Paper**](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/2006%20-%20IEEE_TKDE%20-%20Zhou_Liu.pdf)][[**Code(Java)**](https://github.com/SCI2SUGR/KEEL/blob/master/src/keel/Algorithms/ImbalancedClassification/CSMethods/MLPerceptronBackpropCS/MLPerceptronBackpropCS.java#L49)]** - Training cost-sensitive neural networks with methods addressing the class imbalance problem.

## Deep Learning

#### *Surveys*
<!-- - **Surveys** -->
  
  - A systematic study of the class imbalance problem in convolutional neural networks (2018, 330+ citations) [[**Paper**](https://arxiv.org/pdf/1710.05381.pdf)]
  
  - Survey on deep learning with class imbalance (2019, 50+ citations) [[**Paper**](https://www.researchgate.net/publication/332165523_Survey_on_deep_learning_with_class_imbalance)]
    
    > **NOTE:** a recent comprehensive survey of the class imbalance problem in deep learning.

#### *Hard example mining*
<!-- - **Hard example mining** -->
  
  - Training region-based object detectors with online hard example mining (CVPR 2016, 840+ citations) [[**Paper**](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.pdf)][[**Code**](https://github.com/abhi2610/ohemh)] - In the later phase of NN training, only do gradient back-propagation for "hard examples" (i.e., with large loss value)

#### *Loss function engineering*
<!-- - **Loss function engineering** -->
  
  - Focal loss for dense object detection (ICCV 2017, 2600+ citations) [[**Paper**](http://10.187.70.31/openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)][[**Code (detectron2)**](https://github.com/facebookresearch/detectron2)][[**Code (unofficial)**](https://github.com/clcarwin/focal_loss_pytorch)] - A uniform loss function that focuses training on a sparse set of hard examples to prevents the vast number of easy negatives from overwhelming the detector during training. 
    > **NOTE:** elegant solution, high influence.

  - Training deep neural networks on imbalanced data sets (IJCNN 2016, 110+ citations) [[**Paper**](https://www.researchgate.net/publication/309778930_Training_deep_neural_networks_on_imbalanced_data_sets)] - Mean (square) false error that can equally capture classification errors from both the majority class and the minority class.
  
  - Deep imbalanced attribute classification using visual attention aggregation (ECCV 2018, 30+ citation) [[**Paper**](http://10.187.70.39/openaccess.thecvf.com/content_ECCV_2018/papers/Nikolaos_Sarafianos_Deep_Imbalanced_Attribute_ECCV_2018_paper.pdf)][[**Code**](https://github.com/cvcode18/imbalanced_learning)]
  
  - Imbalanced deep learning by minority class incremental rectification (TPAMI 2018, 60+ citations) [[**Paper**](https://arxiv.org/pdf/1804.10851.pdf)] - Class Rectification Loss for minimizing the dominant effect of majority classes by discovering sparsely sampled boundaries of minority classes in an iterative batch-wise learning process.
  
  - Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss (NIPS 2019, 10+ citations) [[**Paper**](https://papers.nips.cc/paper/8435-learning-imbalanced-datasets-with-label-distribution-aware-margin-loss.pdf)][[**Code**](https://github.com/kaidic/LDAM-DRW)]  - A theoretically-principled label-distribution-aware margin (LDAM) loss motivated by minimizing a margin-based generalization bound.
  
  - Gradient harmonized single-stage detector (AAAI 2019, 40+ citations) [[**Paper**](https://arxiv.org/pdf/1811.05181.pdf)][[**Code**](https://github.com/libuyu/GHM_Detection)] - Compared to Focal Loss, which only down-weights "easy" negative examples, GHM also down-weights "very hard" examples as they are likely to be outliers. 

  - Class-Balanced Loss Based on Effective Number of Samples (CVPR 2019, 70+ citations) [[**Paper**](http://10.187.70.34/openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf)][[**Code**](https://github.com/richardaecn/class-balanced-loss)] - a simple and generic class-reweighting mechanism based on Effective Number of Samples.
  
  - Influence-Balanced Loss for Imbalanced Visual Classification (ICCV 2021) [[**Paper**](https://arxiv.org/pdf/2110.02444.pdf)][[**Code**](https://github.com/pseulki/IB-Loss)]

#### *Meta-learning*
<!-- - **Meta-learning** -->

  - Learning to model the tail (NIPS 2017, 70+ citations) [[**Paper**](http://10.187.70.33/papers.nips.cc/paper/7278-learning-to-model-the-tail.pdf)] - Transfer meta-knowledge from the data-rich classes in the head of the distribution to the data-poor classes in the tail.
  
  - Learning to reweight examples for robust deep learning (ICML 2018, 150+ citations) [[**Paper**](http://10.187.70.24/proceedings.mlr.press/v80/ren18a/ren18a.pdf)][[**Code**](https://github.com/uber-research/learning-to-reweight-examples)] - Implicitly learn a weight function to reweight the samples in gradient updates of DNN.
    > **NOTE:** representative work to solve the class imbalance problem through meta-learning.

  - Meta-weight-net: Learning an explicit mapping for sample weighting (NIPS 2019) [[**Paper**](https://papers.nips.cc/paper/8467-meta-weight-net-learning-an-explicit-mapping-for-sample-weighting.pdf)][[**Code**](https://github.com/xjtushujun/meta-weight-net)] - Explicitly learn a weight function (with an MLP as the function approximator) to reweight the samples in gradient updates of DNN.
  
  - Learning Data Manipulation for Augmentation and Weighting (NIPS 2019) [[**Paper**](https://www.cs.cmu.edu/~zhitingh/data/neurips19_data_manip_preprint.pdf)][[**Code**](https://github.com/tanyuqian/learning-data-manipulation)]
  
  - Learning to Balance: Bayesian Meta-Learning for Imbalanced and Out-of-distribution Tasks (ICLR 2020) [[**Paper**](https://openreview.net/attachment?id=rkeZIJBYvr&name=original_pdf)][[**Code**](https://github.com/haebeom-lee/l2b)]

  - MESA: Boost Ensemble Imbalanced Learning with MEta-SAmpler (NeurIPS 2020) [[**Paper**](https://arxiv.org/pdf/2010.08830.pdf)][[**Code**](https://github.com/ZhiningLiu1998/mesa)][[**Video**](https://studio.slideslive.com/web_recorder/share/20201020T134559Z__NeurIPS_posters__17343__mesa-effective-ensemble-imbal?s=d3745afc-cfcf-4d60-9f34-63d3d811b55f)]
  
    > **NOTE:** meta-learning-powered ensemble learning

#### *Representation Learning*
<!-- - **Representation Learning** -->

  - Learning deep representation for imbalanced classification (CVPR 2016, 220+ citations) [[**Paper**](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Huang_Learning_Deep_Representation_CVPR_2016_paper.pdf)]

  - Supervised Class Distribution Learning for GANs-Based Imbalanced Classification (ICDM 2019) [[**Paper**](https://ieeexplore.ieee.xilesou.top/abstract/document/8970900)]
  
  - Decoupling Representation and Classifier for Long-tailed Recognition (ICLR 2020) [[**Paper**](https://arxiv.org/pdf/1910.09217.pdf)][[**Code**](https://github.com/facebookresearch/classifier-balancing)]
    > **NOTE:** interesting findings on representation learning and classifier learning

#### *Posterior Recalibration*
<!-- - **Posterior Recalibration** -->
  
  - Posterior Re-calibration for Imbalanced Datasets (NeurIPS 2020) [[**Paper**](https://arxiv.org/pdf/2010.11820.pdf)][[**Code**](https://github.com/GT-RIPL/UNO-IC)]

  - Long-tail learning via logit adjustment (ICLR 2021) [[**Paper**](https://arxiv.org/pdf/2007.07314v1.pdf)][[**Code**](https://github.com/google-research/google-research/tree/master/logit_adjustment)]

#### *Semi/Self-supervised Learning*
<!-- - **Semi/Self-supervised Learning** -->

  - Rethinking the Value of Labels for Improving Class-Imbalanced Learning (NeurIPS 2020) [[**Paper**](https://arxiv.org/pdf/2006.07529.pdf)][[**Code**](https://github.com/YyzHarry/imbalanced-semi-self)][[**Video**](https://www.youtube.com/watch?v=XltXZ3OZvyI&feature=youtu.be)]
    
    > **NOTE:** semi-supervised training / self-supervised pre-training helps imbalance learning
  
  - Distribution Aligning Refinery of Pseudo-label for Imbalanced Semi-supervised Learning (NeurIPS 2020) [[**Paper**](https://arxiv.org/pdf/2007.08844.pdf)][[**Code**](https://github.com/bbuing9/DARP)]


#### *Curriculum Learning*
<!-- - **Curriculum Learning** -->

  - Dynamic Curriculum Learning for Imbalanced Data Classification (ICCV 2019) [[**Paper**](http://10.187.70.15/openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Dynamic_Curriculum_Learning_for_Imbalanced_Data_Classification_ICCV_2019_paper.pdf)]

#### *Two-phase Training*
<!-- - **Two-phase Training** -->

  - Brain tumor segmentation with deep neural networks (2017, 1200+ citations) [[**Paper**](https://arxiv.org/pdf/1505.03540.pdf)][[**Code (unofficial)**](https://github.com/naldeborgh7575/brain_segmentation)]

    > Pre-training on balanced dataset, fine-tuning the last output layer before softmax on the original, imbalanced data.

#### *Network Architecture*
<!-- - **Network Architecture** -->

  - BBN: Bilateral-Branch Network with Cumulative Learning for Long-Tailed Visual Recognition (CVPR 2020) [[**Paper**](https://arxiv.org/pdf/1912.02413.pdf)][[**Code**](https://github.com/Megvii-Nanjing/BBN)]
  
  - Class-Imbalanced Deep Learning via a Class-Balanced Ensemble (TNNLS 2021) [[**Paper**](https://ieeexplore.ieee.org/abstract/document/9416240)]

#### *Graph Neural Networks*
<!-- - **Graph Neural Networks** -->

  - GraphSMOTE: Imbalanced Node Classification on Graphs with Graph Neural Networks (WSDM 2021) [[**Paper**](https://dl.acm.org/doi/pdf/10.1145/3437963.3441720)][[**Code**](https://github.com/TianxiangZhao/GraphSmote)]

#### *Deep Generative Model*
<!-- - **Deep Generative Model** -->
  
  - Deep Generative Model for Robust Imbalance Classification (CVPR 2020) [[**Paper**](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Deep_Generative_Model_for_Robust_Imbalance_Classification_CVPR_2020_paper.pdf)]

#### *Imbalanced Regression*
<!-- - **Imbalanced Regression** -->
  
  - Delving into Deep Imbalanced Regression (ICML 2021) [[**Paper**](https://arxiv.org/pdf/2102.09554.pdf)][[**Code**](https://github.com/YyzHarry/imbalanced-regression)][[**Video**](https://www.youtube.com/watch?v=grJGixofQRU)]
  
  - Density-based weighting for imbalanced regression (Machine Learning [J], 2021) [[**Paper**](https://link.springer.com/article/10.1007/s10994-021-06023-5)][[**Code**](https://github.com/SteiMi/density-based-weighting-for-imbalanced-regression)] 

## Anomaly Detection

- **Surveys**

  - Anomaly detection: A survey (ACM computing surveys, 2009, 9000+ citations) [[**Paper**](http://10.187.70.15/cinslab.com/wp-content/uploads/2019/03/xiaorong.pdf)]
  
  - A survey of network anomaly detection techniques (2017, 700+ citations) [[**Paper**](https://www.gta.ufrj.br/~alvarenga/files/CPE826/Ahmed2016-Survey.pdf)]

- **Classification-based**

  - One-class SVMs for document classification (JMLR, 2001, 1300+ citations) [[**Paper**](http://10.187.70.31/www.jmlr.org/papers/volume2/manevitz01a/manevitz01a.pdf)]
  
  - One-class Collaborative Filtering (ICDM 2008, 1000+ citations) [[**Paper**](https://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/04781145.pdf)]
  
  - Isolation Forest (ICDM 2008, 1000+ citations) [[**Paper**](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf?q=isolation-forest)]
  
  - Anomaly Detection using One-Class Neural Networks (2018, 200+ citations) [[**Paper**](https://arxiv.org/pdf/1802.06360.pdf)]
  
  - Anomaly Detection with Robust Deep Autoencoders (KDD 2017, 170+ citations) [[**Paper**](https://pdfs.semanticscholar.org/c112/b06d3dac590b4cc111e5ec9c805d0b086c6e.pdf)]

# Miscellaneous

## Datasets

- **`imbalanced-learn` datasets**

  > This collection of datasets is from [`imblearn.datasets.fetch_datasets`](https://imbalanced-learn.org/stable/references/generated/imblearn.datasets.fetch_datasets.html).

  | ID | Name |	Repository & Target |	Ratio	| #S | #F |
  -|-|-|-|-|-
  | 1	| ecoli 			| UCI, target: imU 				| 8.6:1	| 336	| 7
  | 2	| optical_digits 	| UCI, target: 8 				| 9.1:1	| 5,620	| 64
  | 3	| satimage 			| UCI, target: 4 				| 9.3:1	| 6,435	| 36
  | 4	| pen_digits		| UCI, target: 5 				| 9.4:1	| 10,992| 16
  | 5	| abalone 			| UCI, target: 7 				| 9.7:1	| 4,177	| 10
  | 6	| sick_euthyroid 	| UCI, target: sick euthyroid 	| 9.8:1	| 3,163	| 42
  | 7	| spectrometer 		| UCI, target: > =44 			| 11:1	| 531	| 93
  | 8	| car_eval_34 		| UCI, target: good, v good		| 12:1	| 1,728	| 21
  | 9	| isolet 			| UCI, target: A, B 			| 12:1	| 7,797	| 617
  | 10 |	us_crime	| UCI, target: >0.65			| 12:1 | 1,994 | 100
  | 11 |	yeast_ml8	| LIBSVM, target: 8				| 13:1 | 2,417 | 103
  | 12 |	scene		| LIBSVM, target: >one label	| 13:1 | 2,407 | 294
  | 13 |	libras_move	| UCI, target: 1				| 14:1 | 360   | 90
  | 14 |	thyroid_sick| UCI, target: sick				| 15:1 | 3,772 | 52
  | 15 |	coil_2000	| KDD, CoIL, target: minority 	| 16:1 | 9,822 | 85
  | 16 | arrhythmia		| UCI, target: 06		 | 17:1 | 452	| 278
  | 17 | solar_flare_m0	| UCI, target: M->0		 | 19:1 | 1,389	| 32
  | 18 | oil				| UCI, target: minority	 | 22:1 | 937	| 49
  | 19 | car_eval_4		| UCI, target: vgood	 | 26:1 | 1,728	| 21
  | 20 | wine_quality		| UCI, wine, target: <=4 | 26:1 | 4,898	| 11
  | 21 | letter_img		| UCI, target: Z		 | 26:1 | 20,000| 16
  | 22 | yeast_me2		| UCI, target: ME2		 | 28:1 | 1,484	| 8
  | 23 | webpage			| LIBSVM, w7a, target: minority	 | 33:1 | 34,780| 300
  | 24 | ozone_level		| UCI, ozone, data		 | 34:1 | 2,536	| 72
  | 25 | mammography		| UCI, target: minority	 | 42:1 | 11,183	| 6
  | 26 | protein_homo		| KDD CUP 2004, minority | 111:1 | 145,751	| 74
  | 27 | abalone_19		| UCI, target: 19		 | 130:1 | 4,177	| 10

- **Imbalanced Databases**

  Link: https://github.com/gykovacs/mldb


## Github Repositories

### *Algorithms & Utilities & Jupyter Notebooks*

  - [imbalanced-algorithms](https://github.com/dialnd/imbalanced-algorithms) - Python-based implementations of algorithms for learning on imbalanced data.
  
  - [imbalanced-dataset-sampler](https://github.com/ufoym/imbalanced-dataset-sampler) - A (PyTorch) imbalanced dataset sampler for oversampling low frequent classes and undersampling high frequent ones.
  
  - [class_imbalance](https://github.com/wangz10/class_imbalance) - Jupyter Notebook presentation for class imbalance in binary classification.
  
  - [Multi-class-with-imbalanced-dataset-classification](https://github.com/javaidnabi31/Multi-class-with-imbalanced-dataset-classification) - Perform multi-class classification on imbalanced 20-news-group dataset.
  - [Advanced Machine Learning with scikit-learn: Imbalanced classification and text data](https://github.com/amueller/ml-workshop-4-of-4) - Different approaches to feature selection, and resampling methods for imbalanced data.

### *Paper list*

  - [Anomaly Detection Learning Resources](https://github.com/yzhao062/anomaly-detection-resources) by [yzhao062](https://github.com/yzhao062) - Anomaly detection related books, papers, videos, and toolboxes.

  - [Paper-list-on-Imbalanced-Time-series-Classification-with-Deep-Learning](https://github.com/danielgy/Paper-list-on-Imbalanced-Time-series-Classification-with-Deep-Learning) - Imbalanced Time-series Classification

### *Slides*

  - [acm_imbalanced_learning](https://github.com/timgasser/acm_imbalanced_learning) - slides and code for the ACM Imbalanced Learning talk on 27th April 2016 in Austin, TX.