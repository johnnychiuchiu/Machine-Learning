# Machine-Learning
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/johnnychiuchiu/Machine-Learning/blob/master/LICENSE)

I will continuously update some reproducible machine learning note in R and Python in this repo to record my learning journey in data science.

***

### Projects

A list of end to end machine learning projects. Scopes includes data preprocessing, data visualization, model building, parameter tuning, and result interpretation.
* **Titanic: Machine Learning from Disaster**: predict what sorts of people were likely to survive from the tragedy. [[folder]](https://github.com/johnnychiuchiu/Machine-Learning/tree/master/Projects/titanic)
* **Music Recommender**: build up an end-to-end music recommender application from scratch. [[folder](https://github.com/johnnychiuchiu/Music-Recommender)]
* **Airbnb New User Bookings**: The goal of this project is to help Airbnb predict which country a new user will make his or her first booking. [[folder](https://github.com/bkennedy04/msia420_airbnb_prediction)]
* **Forecasting Energy Consumption**: Predict energy consumption for 200+ buildings using time series data [[folder](https://github.com/johnnychiuchiu/Forecasting-Energy-Consumption)]


***

### Topics

**Design of experiments**
* 2018-01-06 `Steps to conduct A/B Testings and Caveats`[[python nbviewer](http://nbviewer.jupyter.org/github/johnnychiuchiu/Machine-Learning/blob/master/abTesting/abtesting.ipynb)] 
   * Hypothesis Testing | Type I error, Type II error, Power | Determining Sample Size 
* 2018-10-20 `Inferring Causal Effects from Observational Data`[[R nbviewer](http://nbviewer.jupyter.org/github/johnnychiuchiu/Machine-Learning/blob/master/Causal/InferringCausalEffects.ipynb)]
  * Propensity Score Matching | MatchIt(library) | CausalImpact(library)   
* 2019-01-30 `Solving Multi-Armed Bandit Problem through Epsilon-Greedy Algorithm`[[python nbviewer](https://nbviewer.jupyter.org/github/johnnychiuchiu/Machine-Learning/blob/master/abTesting/MultiArmedBandit.ipynb)]
  * Multi-Armed Bandit | Epsilon Greedy Algorithm | Explore & Exploit

**Deep Learning**
* 2018-04-14 `Use Transfer Learning to identify upright or sideways of images`[[python nbviewer](http://nbviewer.jupyter.org/github/johnnychiuchiu/Machine-Learning/blob/master/DeepLearning/transferLearning.ipynb)]
   * Transfer Learning | keras | data augmentation
* 2018-04-14 `Recognizing hand-written digits using neural network`[[python nbviewer](http://nbviewer.jupyter.org/github/johnnychiuchiu/Machine-Learning/blob/master/DeepLearning/mnist.ipynb?flush_cache=true)]
   * Neural Network | MNIST dataset
* 2018-05-15 `Convolutional Neural Network using Keras`[[python nbviewer](http://nbviewer.jupyter.org/github/johnnychiuchiu/Machine-Learning/blob/master/DeepLearning/cnn/CNN.ipynb)]
   * Filter | Padding | Stride | Pooling | Cifer10 dataset | VGG16
* 2019-02-11 `Study Notes on Word Embedding and Word2Vec`[[python nbviewer](https://nbviewer.jupyter.org/github/johnnychiuchiu/Machine-Learning/blob/master/DeepLearning/wordEmbedding.ipynb)]
   * word embedding | word2vec | skip gram | CBOW | text classification

**Text Analytics**
* 2018-04-08 `Text Classification using Naive Bayes`[[python nbviewer](http://nbviewer.jupyter.org/github/johnnychiuchiu/Machine-Learning/blob/master/TextAnalytics/naiveBayesTextClassification.ipynb)]
   * Bernoulli Naive Bayes | Multinomial Naive Bayes | Laplace Smoothing
* 2018-12-29 `Sentiment Analysis for Movie Reviews`[[python nbviewer](http://nbviewer.jupyter.org/github/johnnychiuchiu/Machine-Learning/blob/master/TextAnalytics/SentimentAnalysis.ipynb)]
   * NLP Process | N-gram | TF-IDF | Text Preprocessing | POS Tagging
* 2019-01-29 `Topic Modeling through Latent Dirichlet Allocation`[[python nbviewer](https://nbviewer.jupyter.org/github/johnnychiuchiu/Machine-Learning/blob/master/TextAnalytics/TopicModeling.ipynb)]
   * Latent Dirichlet Allocation | Topic Modeling | gensim

**KNN Based Modeling**
* 2018-03-19 `KNN-Based Modeling`[[R nbviewer](http://nbviewer.jupyter.org/github/johnnychiuchiu/Machine-Learning/blob/master/knn/KNN.ipynb?flush_cache=true)]
   * K-Nearest Neighbors | Local polynomial regression | kernel weighting function

**Customer Lifetime Value**
* 2017-10-23 `Customer Value calculation using RFM` [[python nbviwer](https://nbviewer.jupyter.org/github/johnnychiuchiu/Machine-Learning/blob/master/CustomerLifetimeValue/RFM/RFM.ipynb)]
* 2018-02-27 `Calculating Customer Lifetime Value` [[R nbviwer](http://nbviewer.jupyter.org/github/johnnychiuchiu/Machine-Learning/blob/master/CustomerLifetimeValue/CustomerLifetimeValue.ipynb)]
   * Simple retention model | General retention model | Survival Analysis | Markov Chain, Migration Model
* 2018-04-17 `Calculating Customer Lifetime Value using Markov Chain` [[python nbviewer](http://nbviewer.jupyter.org/github/johnnychiuchiu/Machine-Learning/blob/master/CustomerLifetimeValue/MarkovChain/markovChain.ipynb)]   
   * Markov Chain | Customer Lifetime Value

**Dimension Reduction**
* 2017-12-20 `Principal Component Analysis` [[python jupyter](https://github.com/johnnychiuchiu/Machine-Learning/blob/master/DimensionReduction/PrincipleComponentAnalysis/pca.ipynb)]
   * PCA | eigenvalue & eigenvector

**Optimization Method**
* 2017-12-13 `Gradient Descent` [[R nbviwer](https://nbviewer.jupyter.org/github/johnnychiuchiu/Machine-Learning/blob/master/OptimizationMethod/gradientDescent.ipynb)] 
   * Batch Gradient Descent | Stochastic Gradient Descent
* 2019-01-25 `Optimization and Heuristics` [[python nbviwer](https://nbviewer.jupyter.org/github/johnnychiuchiu/Machine-Learning/blob/master/OptimizationMethod/Optimization%20and%20Heuristics.ipynb)] 
   * Linear Programming | Piecewise Linear Programming | Shadow Price        

**Model Selection Method**
* 2017-12-15 `Model Selection Method` [[python nbviwer](https://nbviewer.jupyter.org/github/johnnychiuchiu/Machine-Learning/blob/master/ModelSelection/modelSelection.ipynb)]
   * Cross Validation | Out of Bag Estimate | Grid Search

**Tree based models**
* 2017-12-11 `Decision Tree Introduction` [[python nbviwer](https://nbviewer.jupyter.org/github/johnnychiuchiu/Machine-Learning/blob/master/EnsembleMethods/decisionTree.ipynb)]
   * Information Gain | Impurity measure | Entropy | Gini Index | Tree Pruning concept
* 2017-12-11 `Bagging and Random Forest` [[python nbviwer](https://nbviewer.jupyter.org/github/johnnychiuchiu/Machine-Learning/blob/master/EnsembleMethods/Bagging/randomForest.ipynb)]
   * Ensemble method | Feature importance | Bagging | Random Forest
* 2017-12-12 `Gradient Boosting Machine for Regression` [[python nbviwer](https://nbviewer.jupyter.org/github/johnnychiuchiu/Machine-Learning/blob/master/EnsembleMethods/Boosting/boostingRegression.ipynb)]
   * Boosting | Gradient Descent | GBRT | Pseudo Residual | MLE
* 2017-12-13 `Gradient Boosting Machine for Classificaiton` [[python nbviwer](https://nbviewer.jupyter.org/github/johnnychiuchiu/Machine-Learning/blob/master/EnsembleMethods/Boosting/boostingClassification.ipynb)]   
   * Boosting | Cross Entropy | Softmax Function 
* 2017-09-11 `xgboost parameter tuning` [[python jupyter](https://nbviewer.jupyter.org/github/johnnychiuchiu/Machine-Learning/blob/master/ExtremeGradientBoosting/xgboost_zillow_home_value.ipynb)]

**Recommender system**
* 2017-09-19 `Understand Collaborative Filtering From Scratch` [[python nbviwer](https://nbviewer.jupyter.org/github/johnnychiuchiu/Machine-Learning/blob/master/RecommenderSystem/collaborative_filtering.ipynb)]
    * User-User CF | Item-Item CF
* 2017-11-24 `Build Up My Own Recommended Song Playlist from Scratch` [[python nbviwer](https://nbviewer.jupyter.org/github/johnnychiuchiu/Machine-Learning/blob/master/RecommenderSystem/latentFactorModel.ipynb)]
   * Latent Factor Model | Alternating Least Squares | Collaborative Filtering

**Regression**
*  2017-11-1 `Linear Regression Model Building Guideline` [[R nbviwer](http://nbviewer.jupyter.org/github/johnnychiuchiu/Machine-Learning/blob/master/LinearRegression/linearRegressionModelBuilding.ipynb)] 
    * Linear Regression | Lasso and Ridge | Model Diagnostics | Model Selection Criterion
* 2017-11-09 `Logistic Regression for binary, nominal, and ordinal response` [[R nbviwer](https://nbviewer.jupyter.org/github/johnnychiuchiu/Machine-Learning/blob/master/LogisticRegression/logisticRegression.ipynb)]
  * Logistic Regression | Maximum probability classifier | Bayes Classifier | ROC, AUC
  
**Clustering**  
* 2017-11-15 `Gaussian Mixture Model` [[python nbviwer](https://nbviewer.jupyter.org/github/johnnychiuchiu/Machine-Learning/blob/master/Clustering/GaussianMixtureModel/gmm.ipynb)]
  * clustering | outlier detection | EM steps | density estimation

**Discriminant Analysis**
 * 2017-11-18 `Discriminant Analysis` [[R nbviwer](https://nbviewer.jupyter.org/github/johnnychiuchiu/Machine-Learning/blob/master/DiscriminantAnalysis/discriminantAnalysis.ipynb)]
   * LDA | QDA | Bayes Classifier



***
### Others
* 2017-12-9 `SQL command note` [[Rmd](https://github.com/johnnychiuchiu/Machine-Learning/blob/master/others/sql_command_note.Rmd)]
* 2018-02-19 `pandas command note` [[nbviwer](http://nbviewer.jupyter.org/github/johnnychiuchiu/Machine-Learning/blob/master/others/pandas_command_note.ipynb)]
* 2018-04-08 `HDFS command note` [[nbviewer](http://nbviewer.jupyter.org/github/johnnychiuchiu/Machine-Learning/blob/master/others/hdfs_command_note.ipynb)]
* 2018-05-20 `spark command note - RDD` [[nbviewer](http://nbviewer.jupyter.org/github/johnnychiuchiu/Machine-Learning/blob/master/others/spark_rdd_note.ipynb)]
* 2018-05-20 `spark command note - DataFrame` [[nbviewer](http://nbviewer.jupyter.org/github/johnnychiuchiu/Machine-Learning/blob/master/others/spark_dataframe_note.ipynb)]
* 2018-06-01 `linux command note` [[nbviewer](http://nbviewer.jupyter.org/github/johnnychiuchiu/Machine-Learning/blob/master/others/linux_command_note.ipynb)]
* 2018-06-01 `python plot note` [[nbviewer](http://nbviewer.jupyter.org/github/johnnychiuchiu/Machine-Learning/blob/master/others/python_plots.ipynb)]
* 2018-06-01 `python command note` [[nbviewer](http://nbviewer.jupyter.org/github/johnnychiuchiu/Machine-Learning/blob/master/others/python_command_note.ipynb)]
* 2018-06-09 `hive command note` [[nbviewer](http://nbviewer.jupyter.org/github/johnnychiuchiu/Machine-Learning/blob/master/others/hive_command_note.ipynb)]
* 2018-06-10 `neo4j- Cypher command note` [[nbviewer](http://nbviewer.jupyter.org/github/johnnychiuchiu/Machine-Learning/blob/master/others/neo4j_command_note.ipynb?flush_cache=true)]
* 2018-06-10 `hbase command note` [[nbviewer](http://nbviewer.jupyter.org/github/johnnychiuchiu/Machine-Learning/blob/master/others/hbase_command_note.ipynb?flush_cache=true)]
* 2018-06-14 `pig command note` [[nbviewer](http://nbviewer.jupyter.org/github/johnnychiuchiu/Machine-Learning/blob/master/others/pig_command_note.ipynb)]
* 2018-10-15 `regular expression note` [[nbviewer](http://nbviewer.jupyter.org/github/johnnychiuchiu/Machine-Learning/blob/master/others/regular_expression_note.ipynb)]



***

### Pending List
* 2017-2-23 `Linear Regression non-traditional model building`.
* 2017-02-19 `Random Forest for classification problems`
* 2017-03-01 `extreme gradient boosting for classification problems`
* 2017-03-15 `gradient boosting tree for classification problems`
* 2017-04-08 `using extreme gradient boosting to solve` [predicting-red-hat-business-value problem from kaggle](https://www.kaggle.com/c/predicting-red-hat-business-value)



