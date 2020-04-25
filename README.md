# Awesome Imbalanced Learning

[![Awesome](https://awesome.re/badge-flat.svg)](https://awesome.re)

A curated list of awesome imbalanced learning papers, codes, frameworks and libraries. 

**Class-imbalance** (also known as the long-tail problem) is the fact that the classes are not represented equally in a classification problem, which is quite common in practice. For instance, fraud detection, prediction of rare adverse drug reactions and prediction gene families. Failure to account for the class imbalance often causes inaccurate and decreased predictive performance of many classification algorithms. Imbalanced learning aims to tackle the class imbalance problem to learn an unbiased model from imbalanced data.

Inspired by [awesome-machine-learning](https://github.com/josephmisiti/awesome-machine-learning). Contributions are welcomed!

> _Items marked with :accept: are personally recommended (important/high-quality papers or libraries)._

# Table of Contents

- [Awesome Imbalanced Learning](#awesome-imbalanced-learning)
- [Table of Contents](#table-of-contents)
- [Libraries](#libraries)
    - [Python](#python)
    - [R](#r)
    - [Java](#java)
    - [Scalar](#scalar)
    - [Julia](#julia)
- [Papers](#papers)
  - [Surveys](#surveys)
  - [Deep Learning](#deep-learning)
  - [Data resampling](#data-resampling)
  - [Cost-sensitive Learning](#cost-sensitive-learning)
  - [Ensemble Learning](#ensemble-learning)
  - [Anomaly Detection](#anomaly-detection)
- [Others](#others)
  - [Imbalanced Datasets](#imbalanced-datasets)
  - [Other Resources](#other-resources)


# Libraries

### Python
- [**imbalanced-learn**](https://imbalanced-learn.org/stable/) [[Github](https://github.com/scikit-learn-contrib/imbalanced-learn)][[Documentation](https://imbalanced-learn.readthedocs.io/en/stable/)][[Paper](http://10.187.70.34/www.jmlr.org/papers/volume18/16-365/16-365.pdf)] - imbalanced-learn is a python package offering a number of ***re-sampling*** techniques commonly used in datasets showing strong between-class imbalance. It is compatible with [scikit-learn](https://scikit-learn.org/stable/) and is part of [scikit-learn-contrib](https://github.com/scikit-learn-contrib) projects. 
    > :accept: written in python, easy to use.

### R
- [**smote_variants**](https://smote-variants.readthedocs.io/en/latest/) [[Documentation](https://smote-variants.readthedocs.io/en/latest/)][[Github](https://github.com/analyticalmindsltd/smote_variants)] - A collection of 85 minority ***over-sampling*** techniques for imbalanced learning with multi-class oversampling and model selection features (support R and Julia).
- [**caret**](https://cran.r-project.org/web/packages/caret/index.html) [[Documentation](http://topepo.github.io/caret/index.html)][[Github](https://github.com/topepo/caret)] - Contains the implementation of Random under/over-sampling.
- [**ROSE**](https://cran.r-project.org/web/packages/ROSE/index.html) [[Documentation](https://www.rdocumentation.org/packages/ROSE/versions/0.0-3)] - Contains the implementation of [ROSE](https://journal.r-project.org/archive/2014-1/menardi-lunardon-torelli.pdf) (Random Over-Sampling Examples).
- [**DMwR**](https://cran.r-project.org/web/packages/DMwR/index.html) [[Documentation](https://www.rdocumentation.org/packages/DMwR/versions/0.4.1)] - Contains the implementation of [SMOTE](https://arxiv.org/pdf/1106.1813.pdf) (Synthetic Minority Over-sampling TEchnique).

### Java
- [**KEEL**](https://sci2s.ugr.es/keel/description.php) [[Github](https://github.com/SCI2SUGR/KEEL)][[Paper](https://sci2s.ugr.es/sites/default/files/ficherosPublicaciones/0758_Alcalaetal-SoftComputing-Keel1.0.pdf)] - KEEL provides a simple ***GUI based*** on data flow to design experiments with different datasets and computational intelligence algorithms (***paying special attention to evolutionary algorithms***) in order to assess the behavior of the algorithms. This tool includes many widely used imbalanced learning techniques such as (evolutionary) over/under-resampling, cost-sensitive learning, algorithm modification, and ensemble learning methods. 
    > :accept: wide variety of classical classification, regression, preprocessing algorithms included.

### Scalar
- [**undersampling**](https://github.com/NestorRV/undersampling) [[Documentation](https://nestorrv.github.io/)][[Github](https://github.com/NestorRV/undersampling)] - A Scala library for ***under-sampling and their ensemble variants*** in imbalanced classification.

### Julia
- [**smote_variants**](https://smote-variants.readthedocs.io/en/latest/) [[Documentation](https://smote-variants.readthedocs.io/en/latest/)][[Github](https://github.com/analyticalmindsltd/smote_variants)] - A collection of 85 minority ***over-sampling*** techniques for imbalanced learning with multi-class oversampling and model selection feature (support R and Julia).

# Papers

## Surveys

- [Learning from imbalanced data](https://www.sci-hub.shop/10.1109/tkde.2008.239) (2009, 4700+ citations) - Highly cited, classic survey paper. It systematically reviewed the popular solutions, evaluation metrics, and challenging problems in future research in this area (as of 2009). 
    > :accept: classic work.
- [Learning from imbalanced data: open challenges and future directions](https://www.researchgate.net/publication/301596547_Learning_from_imbalanced_data_Open_challenges_and_future_directions) (2016, 400+ citations) - This paper concentrates on discussing the open issues and challenges in imbalanced learning, such as extreme class imbalance, dealing imbalance in online/stream learning, multi-class imbalanced learning, and semi/un-supervised imbalanced learning.
- [Learning from class-imbalanced data: Review of methods and applications](https://www.researchgate.net/publication/311977198_Learning_from_class-imbalanced_data_Review_of_methods_and_applications) (2017, 400+ citations) - A recent exhaustive survey of imbalanced learning methods and applications, a total of 527 papers were included in this study. It provides several detailed taxonomies of existing methods and also the recent trend of this research area.
    > :accept: a systematic survey with detailed taxonomies of existing methods.

## Deep Learning

- **Surveys**
  - [A systematic study of the class imbalance problem in convolutional neural networks](https://arxiv.org/pdf/1710.05381.pdf) (2018, 330+ citations)
  - [Survey on deep learning with class imbalance](https://www.researchgate.net/publication/332165523_Survey_on_deep_learning_with_class_imbalance) (2019, 50+ citations)
    > :accept: a recent comprehensive survey of the class imbalance problem in deep learning.

- **Hard example mining**
  - [Training region-based object detectors with online hard example mining](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.pdf) (CVPR 2016, 840+ citations) - In the later phase of NN training, only do gradient back-propagation for "hard examples" (i.e., with large loss value)

- **Loss function engineering**
  - [Training deep neural networks on imbalanced data sets](https://www.researchgate.net/publication/309778930_Training_deep_neural_networks_on_imbalanced_data_sets) (IJCNN 2016, 110+ citations) - Mean (square) false error that can equally capture classification errors from both the majority class and the minority class.
  - [Focal loss for dense object detection](http://10.187.70.31/openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf) [[Code (Unofficial)](https://github.com/clcarwin/focal_loss_pytorch)] (ICCV 2017, 2600+ citations) - A uniform loss function that focuses training on a sparse set of hard examples to prevents the vast number of easy negatives from overwhelming the detector during training. 
    > :accept: elegant solution, high influence.
  - [Deep imbalanced attribute classification using visual attention aggregation](http://10.187.70.39/openaccess.thecvf.com/content_ECCV_2018/papers/Nikolaos_Sarafianos_Deep_Imbalanced_Attribute_ECCV_2018_paper.pdf) [[Code](https://github.com/cvcode18/imbalanced_learning)] (ECCV 2018, 30+ citation)
  - [Imbalanced deep learning by minority class incremental rectification](https://arxiv.org/pdf/1804.10851.pdf) (TPAMI 2018, 60+ citations) - Class Rectification Loss for minimizing the dominant effect of majority classes by discovering sparsely sampled boundaries of minority classes in an iterative batch-wise learning process.
  - [Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss](https://papers.nips.cc/paper/8435-learning-imbalanced-datasets-with-label-distribution-aware-margin-loss.pdf) [[Code](https://github.com/kaidic/LDAM-DRW)] (NIPS 2019, 10+ citations) - A theoretically-principled label-distribution-aware margin (LDAM) loss motivated by minimizing a margin-based generalization bound.
  - [Gradient harmonized single-stage detector](https://arxiv.org/pdf/1811.05181.pdf) [[Code](https://github.com/libuyu/GHM_Detection)] (AAAI 2019, 40+ citations) - Compared to Focal Loss, which only down-weights "easy" negative examples, GHM also down-weights "very hard" examples as they are likely to be outliers. 
    > :accept: interesting idea: harmonizing the contribution of examples on the basis of their gradient distribution.
  - [Class-Balanced Loss Based on Effective Number of Samples](http://10.187.70.34/openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf) (CVPR 2019, 70+ citations) - a simple and generic class-reweighting mechanism based on Effective Number of Samples.

- **Meta-learning**
  - [Learning to model the tail](http://10.187.70.33/papers.nips.cc/paper/7278-learning-to-model-the-tail.pdf) (NIPS 2017, 70+ citations) - Transfer meta-knowledge from the data-rich classes in the head of the distribution to the data-poor classes in the tail.
  - [Learning to reweight examples for robust deep learning](http://10.187.70.24/proceedings.mlr.press/v80/ren18a/ren18a.pdf) [[Code](https://github.com/uber-research/learning-to-reweight-examples)] (ICML 2018, 150+ citations) - Implicitly learn a weight function to reweight the samples in gradient updates of DNN.
    > :accept: representative work to solve the class imbalance problem through meta-learning.
  - [Meta-weight-net: Learning an explicit mapping for sample weighting](https://papers.nips.cc/paper/8467-meta-weight-net-learning-an-explicit-mapping-for-sample-weighting.pdf) [[Code](https://github.com/xjtushujun/meta-weight-net)] (NIPS 2019) - Explicitly learn a weight function (with an MLP as the function approximator) to reweight the samples in gradient updates of DNN.
  - [Learning Data Manipulation for Augmentation and Weighting](https://www.cs.cmu.edu/~zhitingh/data/neurips19_data_manip_preprint.pdf) [[Code](https://github.com/tanyuqian/learning-data-manipulation)] (NIPS 2019)
  - [Learning to Balance: Bayesian Meta-Learning for Imbalanced and Out-of-distribution Tasks](https://openreview.net/attachment?id=rkeZIJBYvr&name=original_pdf) [[Code](https://github.com/haebeom-lee/l2b)] (ICLR 2020)

- **Representation Learning**
  - [Learning deep representation for imbalanced classification](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Huang_Learning_Deep_Representation_CVPR_2016_paper.pdf) (CVPR 2016, 220+ citations)
  - [Supervised Class Distribution Learning for GANs-Based Imbalanced Classification](https://ieeexplore.ieee.xilesou.top/abstract/document/8970900) (ICDM 2019)
  - [Decoupling Representation and Classifier for Long-tailed Recognition](https://arxiv.org/pdf/1910.09217.pdf) (ICLR 2020)

- **Curriculum learning**
  - [Dynamic Curriculum Learning for Imbalanced Data Classification](http://10.187.70.15/openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Dynamic_Curriculum_Learning_for_Imbalanced_Data_Classification_ICCV_2019_paper.pdf) (ICCV 2019)

- **Two-phase training**
  - [Brain tumor segmentation with deep neural networks](https://arxiv.org/pdf/1505.03540.pdf) (2017, 1200+ citations) - Pre-training on balanced dataset, fine-tuning the last output layer before softmax on the original, imbalanced data.

## Data resampling

- **Over-sampling**
  - ROS [[Code](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/over_sampling/_random_over_sampler.py)] - Random Over-sampling 
  - [SMOTE](https://arxiv.org/pdf/1106.1813.pdf) [[Code](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/over_sampling/_smote.py#L36)] (2002, 9800+ citations) - Synthetic Minority Over-sampling TEchnique 
    > :accept: classic work.
  - [Borderline-SMOTE](https://sci2s.ugr.es/keel/keel-dataset/pdfs/2005-Han-LNCS.pdf) [[Code](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/over_sampling/_smote.py#L220)] (2005, 1400+ citations) - Borderline-Synthetic Minority Over-sampling TEchnique 
  - [ADASYN](https://sci2s.ugr.es/keel/pdf/algorithm/congreso/2008-He-ieee.pdf) [[Code](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/over_sampling/_adasyn.py)] (2008, 1100+ citations) - ADAptive SYNthetic Sampling
  - [SPIDER](https://sci2s.ugr.es/keel/pdf/algorithm/congreso/stefanowski_selective_2008.pdf) [[Code (Java)](https://github.com/SCI2SUGR/KEEL/blob/master/src/keel/Algorithms/ImbalancedClassification/Resampling/SPIDER/SPIDER.java#L57)] (2008, 150+ citations) - Selective Preprocessing of Imbalanced Data
  - [Safe-Level-SMOTE](http://10.187.70.30/150.214.190.154/keel/keel-dataset/pdfs/2009-Bunkhumpornpat-LNCS.pdf) [[Code (Java)](https://github.com/SCI2SUGR/KEEL/blob/master/src/keel/Algorithms/ImbalancedClassification/Resampling/Safe_Level_SMOTE/Safe_Level_SMOTE.java#L58)] (2009, 370+ citations) - Safe Level Synthetic Minority Over-sampling TEchnique
  - [SVM-SMOTE](http://10.187.70.39/ousar.lib.okayama-u.ac.jp/files/public/1/19617/20160528004522391723/IWCIA2009_A1005.pdf) [[Code](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/over_sampling/_smote.py#L417)] (2009, 120+ citations) - SMOTE based on Support Vectors of SVM 
  - [SMOTE-IPF](https://sci2s.ugr.es/sites/default/files/ficherosPublicaciones/1824_2015-INS-Saez.pdf) (2015, 180+ citations) - SMOTE with Iterative-Partitioning Filter
  
- **Under-sampling**
  - RUS [[Code](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/under_sampling/_prototype_selection/_random_under_sampler.py)] - Random Under-sampling 
  - [CNN](https://pdfs.semanticscholar.org/7c37/71fd6829630cf450af853df728ecd8da4ab2.pdf?_ga=2.137274553.882046879.1583413150-1712662047.1583413150) [[Code](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/under_sampling/_prototype_selection/_condensed_nearest_neighbour.py)] (1968, 2100+ citations) - Condensed Nearest Neighbor 
  - [ENN](https://sci2s.ugr.es/keel/dataset/includes/catImbFiles/1972-Wilson-IEEETSMC.pdf) [[Code](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/under_sampling/_prototype_selection/_edited_nearest_neighbours.py)] (1972, 1500+ citations) - Edited Condensed Nearest Neighbor 
  - [TomekLink](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4309452) [[Code](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/under_sampling/_prototype_selection/_tomek_links.py)] (1976, 870+ citations) - Tomek's modification of Condensed Nearest Neighbor 
  - [NCR](https://sci2s.ugr.es/keel/pdf/algorithm/congreso/2001-Laurikkala-LNCS.pdf) [[Code](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/under_sampling/_prototype_selection/_neighbourhood_cleaning_rule.py)] (2001, 500+ citations) - Neighborhood Cleaning Rule 
  - [NearMiss-1 & 2 & 3](https://sci2s.ugr.es/keel/pdf/specific/congreso/jzhang.pdf) [[Code](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/under_sampling/_prototype_selection/_nearmiss.py)] (2003, 420+ citations) - Several kNN approaches to unbalanced data distributions. 
  - [CNN with TomekLink](https://storm.cis.fordham.edu/~gweiss/selected-papers/batista-study-balancing-training-data.pdf) [[Code (Java)](https://github.com/SCI2SUGR/KEEL/blob/master/src/keel/Algorithms/ImbalancedClassification/Resampling/CNN_TomekLinks/CNN_TomekLinks.java#L58)] (2004, 2000+ citations) - Condensed Nearest Neighbor + TomekLink
  - [OSS](https://sci2s.ugr.es/keel/pdf/specific/congreso/kubat97addressing.pdf) [[Code](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/under_sampling/_prototype_selection/_one_sided_selection.py)] (2007, 2100+ citations) - One Side Selection 
  - [EUS](https://www.mitpressjournals.org/doi/pdfplus/10.1162/evco.2009.17.3.275) (2009, 290+ citations) - Evolutionary Under-sampling
  - [IHT](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.649.8727&rep=rep1&type=pdf) [[Code](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/under_sampling/_prototype_selection/_instance_hardness_threshold.py)] (2014, 130+ citations) - Instance Hardness Threshold 
  
- **Hybrid-sampling**
  - [SMOTE-Tomek & SMOTE-ENN](http://10.187.70.37/150.214.190.154/keel/dataset/includes/catImbFiles/2004-Batista-SIGKDD.pdf) (2004, 2000+ citations) [[Code (SMOTE-Tomek)](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/combine/_smote_tomek.py)] [[Code (SMOTE-ENN)](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/combine/_smote_enn.py)] - Synthetic Minority Over-sampling TEchnique + Tomek's modification of Condensed Nearest Neighbor/Edited Nearest Neighbor 
    > :accept: extensive experimental evaluation involving 10 different over/under-sampling methods.
  - [SMOTE-RSB](https://sci2s.ugr.es/sites/default/files/ficherosPublicaciones/1434_2012-Ramentol-KAIS.pdf) (2012, 210+ citations) - Hybrid Preprocessing using SMOTE and Rough Sets Theory

## Cost-sensitive Learning

- [CSC4.5](https://www.sci-hub.shop/10.1109/tkde.2002.1000348) [[Code (Java)](https://github.com/SCI2SUGR/KEEL/blob/master/src/keel/Algorithms/ImbalancedClassification/CSMethods/C45CS/C45CS.java#L48)] (2002, 420+ citations) - An instance-weighting method to induce cost-sensitive trees
- [CSSVM](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/2009-Chawla-IEEE_TSMCB-svm-imbalance.pdf) [[Code (Java)](https://github.com/SCI2SUGR/KEEL/blob/master/src/keel/Algorithms/ImbalancedClassification/CSMethods/C_SVMCost/svmClassifierCost.java#L60)] (2008, 710+ citations) - Cost-sensitive SVMs for highly imbalanced classification
- [CSNN](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/2006%20-%20IEEE_TKDE%20-%20Zhou_Liu.pdf) [[Code (Java)](https://github.com/SCI2SUGR/KEEL/blob/master/src/keel/Algorithms/ImbalancedClassification/CSMethods/MLPerceptronBackpropCS/MLPerceptronBackpropCS.java#L49)] (2005, 950+ citations) - Training cost-sensitive neural networks with methods addressing the class imbalance problem.

## Ensemble Learning

- **Boosting-based**
  - [AdaBoost](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/1997-JCSS-Schapire-A%20Decision-Theoretic%20Generalization%20of%20On-Line%20Learning%20(AdaBoost).pdf) [[Code](https://github.com/scikit-learn/scikit-learn/blob/95d4f0841/sklearn/ensemble/_weight_boosting.py#L285)] (1995, 18700+ citations) - Adaptive Boosting with C4.5
  - [DataBoost](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/2004-SIGKDD-GuoViktor.pdf) (2004, 570+ citations) - Boosting with Data Generation for Imbalanced Data
  - [SMOTEBoost](https://sci2s.ugr.es/keel/pdf/algorithm/congreso/2003-PKDD-SMOTEBoost-ChawlaLazarevicHallBowyer.pdf) [[Code](https://github.com/dialnd/imbalanced-algorithms/blob/master/smote.py#L94)] (2003, 1100+ citations) - Synthetic Minority Over-sampling TEchnique Boosting 
    > :accept: classic work.
  - [MSMOTEBoost](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/2011-IEEE%20TSMC%20partC-%20GalarFdezBarrenecheaBustinceHerrera.pdf) (2011, 1300+ citations) - Modified Synthetic Minority Over-sampling TEchnique Boosting
  - [RAMOBoost](https://www.ele.uri.edu/faculty/he/PDFfiles/ramoboost.pdf) [[Code](https://github.com/dialnd/imbalanced-algorithms/blob/master/ramo.py#L133)] (2010, 140+ citations) - Ranked Minority Over-sampling in Boosting 
  - [RUSBoost](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/2010-IEEE%20TSMCpartA-RUSBoost%20A%20Hybrid%20Approach%20to%20Alleviating%20Class%20Imbalance.pdf) [[Code](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/ensemble/_weight_boosting.py#L21)] (2009, 850+ citations) - Random Under-Sampling Boosting 
    > :accept: classic work.
  - [AdaBoostNC](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/2012-wang-IEEE_SMC_B.pdf) (2012, 350+ citations) - Adaptive Boosting with Negative Correlation Learning
  - [EUSBoost](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/2013-galar-PR.pdf) (2013, 210+ citations) - Evolutionary Under-sampling in Boosting

- **Bagging-based**
  - [Bagging](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/1996-ML-Breiman-Bagging%20Predictors.pdf) [[Code](https://github.com/scikit-learn/scikit-learn/blob/95d4f0841/sklearn/ensemble/_bagging.py#L433)] (1996, 23100+ citations) - Bagging predictors 
  - [OverBagging & UnderOverBagging & SMOTEBagging & MSMOTEBagging](https://sci2s.ugr.es/keel/pdf/algorithm/congreso/2009-IEEE%20CIDM-WangYao.pdf) [[Code (SMOTEBagging)](https://github.com/ZhiningLiu1998/self-paced-ensemble/blob/master/canonical_ensemble.py#L615)] (2009, 290+ citations) - Random Over-sampling / Random Hybrid Resampling / SMOTE / Modified SMOTE with Bagging
  - [UnderBagging](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/2003-PAA-%20New%20Applications%20of%20Ensembles%20of%20Classifiers.pdf) [[Code](https://github.com/ZhiningLiu1998/self-paced-ensemble/blob/master/canonical_ensemble.py#L665)] (2003, 170+ citations) - Random Under-sampling with Bagging 

- **Other forms of ensemble**
  - [EasyEnsemble & BalanceCascade](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/2009-IEEE%20TSMCpartB%20Exploratory%20Undersampling%20for%20Class%20Imbalance%20Learning.pdf) [[Code (EasyEnsemble)](https://github.com/scikit-learn-contrib/imbalanced-learn/blob/master/imblearn/ensemble/_easy_ensemble.py#L30)] [[Code (BalanceCascade)](https://github.com/ZhiningLiu1998/self-paced-ensemble/blob/master/canonical_ensemble.py#L709)] (2008, 1300+ citations) - Parallel ensemble training with RUS (EasyEnsemble) / Cascade ensemble training with RUS while iteratively drops well-classified examples (BalanceCascade) 
    > :accept: simple but effective solution.
  - [Self-paced Ensemble](https://arxiv.org/pdf/1909.03500.pdf) [[Code](https://github.com/ZhiningLiu1998/self-paced-ensemble)] (ICDE 2020) - Training Effective Ensemble on Imbalanced Data by Self-paced Harmonizing Classification Hardness 
    > :accept: high performance & computational efficiency & widely applicable to different classifiers.

## Anomaly Detection

- Surveys
  - [Anomaly detection: A survey](http://10.187.70.15/cinslab.com/wp-content/uploads/2019/03/xiaorong.pdf) (2009, 7300+ citations)
  - [A survey of network anomaly detection techniques](https://www.gta.ufrj.br/~alvarenga/files/CPE826/Ahmed2016-Survey.pdf) (2017, 210+ citations)

- Classification-based
  - [One-class SVMs for document classification](http://10.187.70.31/www.jmlr.org/papers/volume2/manevitz01a/manevitz01a.pdf) (2001, 1300+ citations)
  - [One-class Collaborative Filtering](https://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/04781145.pdf) (2008, 830+ citations)
  - [Isolation Forest](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf?q=isolation-forest) (2008, 1000+ citations)
  - [Anomaly Detection using One-Class Neural Networks](https://arxiv.org/pdf/1802.06360.pdf) (2018, 70+ citations)
  - [Anomaly Detection with Robust Deep Autoencoders](https://pdfs.semanticscholar.org/c112/b06d3dac590b4cc111e5ec9c805d0b086c6e.pdf) (KDD 2017, 170+ citations)

# Others

## Imbalanced Datasets

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

Note: This collection of datasets is from [imblearn.datasets.fetch_datasets](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.datasets.fetch_datasets.html#imblearn.datasets.fetch_datasets).

## Other Resources
- [Paper-list-on-Imbalanced-Time-series-Classification-with-Deep-Learning](https://github.com/danielgy/Paper-list-on-Imbalanced-Time-series-Classification-with-Deep-Learning)
- [acm_imbalanced_learning](https://github.com/timgasser/acm_imbalanced_learning) - slides and code for the ACM Imbalanced Learning talk on 27th April 2016 in Austin, TX.
- [imbalanced-algorithms](https://github.com/dialnd/imbalanced-algorithms) - Python-based implementations of algorithms for learning on imbalanced data.
- [imbalanced-dataset-sampler](https://github.com/ufoym/imbalanced-dataset-sampler) - A (PyTorch) imbalanced dataset sampler for oversampling low frequent classes and undersampling high frequent ones.
- [class_imbalance](https://github.com/wangz10/class_imbalance) - Jupyter Notebook presentation for class imbalance in binary classification.
