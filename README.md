# Titanic - Machine Learning from Disaster

## Introduction

This repo holds part of my work on the famous Kaggle competition "Titanic - Machine Learning from Disaster".

As this is an introductory Kaggle competition and being a ML novice myself, please note:

* This is a ML playground to get me familiar with both Python and the basics of ML and classification problems in particular. This work may contain errors in any area, including code, modelling, and even general approaches. I am already aware of some of them and I will aim to fix them in future projects! 
* Adhering to ML best practices was deemed more important than the final score. Therefore, known techniques that can boost Kaggle score were avoided (e.g. leaking test data into train data when imputing missing values).
* I did not hesitate to try out (and submit) multiple models, based on different takes on the EDA/preprocessing process. The intention was to see the effect of adding new features, try out a few different algorithms and observe the effect of different feature scaling and encoding techniques. All this at the cost of possibly overfitting to the Kaggle test set.
* Notebooks may be more verbose than required.

## Project Description

*The competition is simple: use machine learning to create a model that predicts which passengers survived the Titanic shipwreck.*

Full details of the project can be found [here](https://www.kaggle.com/c/titanic/overview).

## Data Description

Data description can be found [here](https://www.kaggle.com/c/titanic/data).

## High level approach

Conceptually, the project pipeline is split into 3 parts. 

1. Exploratory data analysis (EDA)
2. Model building and evaluation
3. Use the final model to predict and submit

This corresponds to the file hierarchy in this repo, which is along the lines of:

**EDA notebooks**

	\Titanic - EDA#.ipynb

**Data files**

	\data\test.csv (original test set)
	\data\test_prepd#.csv (post processed test set)
	\data\train.csv (original train set)
	\data\train_prepd#.csv (post processed train set)

**Model building notebooks**

	\Titanic - <model>#.ipynb

**Final model building and submission files**

	\subs\gender_submission.csv (default Kaggle prediction file)
	\subs\Titanic - Submission - <model># (final model building)
	\subs\submission_<model>#.csv (prediction file for submission)
	
The # represents the EDA iteration - the exploratory data analysis was carried out several times and each iteration led to a different set of models and corresponding submissions.

### Exploratory Data Analysis

As a general principle, the test set was treated as a set of never seen before observations. So, caution was taken, in order to not leak any information from the testing set into the training set. Thus, the test set was not explored, but got recoded and had its missing values imputed.

The process I followed consists of two broad steps:

1. Data exploration

Understand dataset shape and properties, explore features and correlations with visualizations. Investigate for useful data recoding and transformations. Identify missing values.

2. Data Preprocessing / Transformations

Start again from the original file and perform feature recoding, feature extraction, missing value imputation, and all other transformations that are not model specific (e.g. standardization). Also drop unneeded features. Output the processed csv that is going to be used in model building.

In the first iteration of EDA:

* Dropped PassengerID, Ticket
* Created the "Family size" feature by combining the # of siblings/spouses and # of parents/children features
* Created the "Title" feature by extracting the persons honorific from their name.
* Created a binary feature holding the information that the cabin no. is missing.
* Recoded categorical variables into dummies, dropping the first level.
* Imputed all other missing values using a KNN imputer.

In further iterations:

EDA2: 

* Nicer format and structure. Minor improvements in several areas.
* Dummy variables noe do not drop the first level.
* Extracted the first letter of "Cabin" feature and recoded into a new variable named "Deck".
* Used different missing value imputation strategies based around median values of groups.

EDA3:

* Dummies now drop the first level.

### Model building and evaluation

Each iteration of the EDA led to the development of several models. The following models were built:

* A Decision Tree
* A Random Forest Classifier 
* A Support Vector Machine
* A Logistic Regression
* An XGBoost model

For each of the models, the flow of the training and evaluation process is the following:

1. Import the preprocessed training dataset, which is the output of the EDA phase.
2. Perform model specific transformations (e.g. standardization for logistic regression models).
3. Build a preliminary model and assess it on a train/test split and additionally using cross validation.
4. Tune the model hyperparameters using GridSearchCV and/or RandomizedSearchCV on the entire train dataset.
5. Assess the tuned model on a train/test split and additionally using cross validation.

A drawback of this approach is that, since hyperparameter tuning is performed on the entire training dataset, there are no means of getting an estimate of the generalization accuracy (i.e. there is no holdout test set in the training to do a final check of the best model). The answer to this is nested K-fold CV (see Further Work / Stuff to Improve section).

### Final model building and submission

In this step a model has already been picked from the model building step. The chosen model is fit on the entire preprocessed train set and predicts the response of the preprocessed test set. An ensemble model was also built.

A csv file is output to submit to Kaggle.

## Further work / Stuff to improve

* Use nested K-fold cross validation. This would be along the lines of:
	1. Split the training set into train/validation/test.
	2. Use nested K-fold CV to get the generalization accuracy.
		* Each inner run will find the best model on the test/validation split
		* Each outer run will find the generalization accuracy estimate of the best model on the test split
	3. Each outer run of the CV will have a best model. Out of them pick one, pick the best one or use all of them in an ensemble.
	4. Retrain this model on the entire starting train set.
	5. Predict on the test set.
* Use sklearn pipelines. This would also solve information leakage in cases where standardization/normalization is performed within CV.
* Make the KNN imputer impute values on the test set while fit on the train set.
* Drop additional features to reduce multicollinearity and overfitting.
* Use different models and ensembles.
* Explore and recode more features, such as the ticket feature.
* Do further work in random forest models and consider additional hyperparameters and performance metrics.
* Use early stopping in XGBoost models.
