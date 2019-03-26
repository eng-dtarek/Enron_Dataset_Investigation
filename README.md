# Enron Dataset Investigation

## Summary

The goal of the project is to build a prediction model to identify POI from non POI in Enron dataset given some features using Machine learning algorithmsto detect the underlying pattens for POI from the given features.


**Dataset Overview**

There are 146 data points on the data set with 21 features for each person. 18 POIs in the dataset using the names file (poi_names.txt).For nearly every person in the dataset, not every feature has a value denoted with NaN except for POI which has either true or false in its value for all the data points.95 persons have a quantified salary and 111 persons have known email address. (about 14%) of the people in the dataset don't have total_payments filled in. 0% of POI's don't have total_payments filled.

**Outliers**

Using salary and bonus features to detect outliers in the dataset reveals one outlier [Total] which must be removed because it is a spreadsheet quirk.after removing [Total] there were almost 4 more outliers including LAY KENNETH L and SKILLING JEFFREY K which seem to be valid data points so we should leave them in.

## Features Selection

I have used SelectKBest to select the best 5 features to be used in my POI identifier and here is a list of the best 5 features with their scores.

- salary: 14.45
- shared_receipt_with_poi: 9.31
- deferred_income: 7.36
- restricted_stock: 10.11
- total_payments: 62.94

I didn’t have to do feature scaling in my final work because I used Gaussian naive bayes algorithm which is not affected by feature scaling but I used it with SVM (without feature scaling, all the predictions were NON POI [0] resulted in an error on tester.py "Got a divide by zero").

I think that the financial data reveals the POI so I have added a new feature (total_payments_and_stock) represents person’s total payments and stock value (total_payments + total_stock_value) which has an effect on both accuracy and validations.

- Before adding total_payments_and_stock feature accuracy: 0.84, recall: 0.23, precision: 0.35

- After adding total_payments_and_stock feature accuracy: 0.85, recall: 0.32, precision: 0.43

I have chosen 5 an a parameter to SelectKBest because I have tested other numbers like 10, and 15. 5 almost provides the best result when used with Gaussian naive bayes algorithm as shown on the following section.

**Previous work on this section**

I have started by passing 5 as a parameter for SelectKBest to get the best 5 features and the results was as below.

- salary: 14.45
- shared_receipt_with_poi: 9.31
- loan_advances: 8066.59
- restricted_stock: 10.11
- total_payments: 62.94

Using these features with gaussian naive bayes algorithm results in the following performance.

- accuracy: 0.74
- precision: 0.10
- recall: 0.11

The score of "loan_advances" seems weird for me it is very high compared to the other features. So I decided to check it manually on the pdf file. I found that almost all persons have no value for this feature except 2 or 3 persons So I tried removing it from the total 21 features and rerun again SelectKBest to get the best 5 features excluding loan_advances and the performance improved very well.

In SelectKBest parameters I have tested many numbers like 5, 10 and 15 with gaussian naive bayes algorithm to test algorithm’s performance on different groups of features.here is the result.

- best 5 features. accuracy: 0.84, precision: 0.35, recall: 0.23
- best 10 features. accuracy: 0.83, precision: 0.33, recall: 0.25
- best 5 features. accuracy: 0.83, precision: 0.33, recall: 0.26

Passing the best 5 features to gaussian naive bayes algorithm results in the best accuracy and precision.in recall value is 0.03 lower than the best one here (using 15 features)which I think we can ignore in opposite to the values of both accuracy and precision and definitely the cost of training with extra 10 features.

## Machine Learning Algorithms

Three classification algorithms were tested to identify POI:

- Gaussian Naive Bayes. accuracy: 0.85, precision: 0.43, recall: 0.32 
- Decision Tree. accuracy: 0.87, precision: 0.59, recall: 0.12
- SVM. accuracy: 0.87, precision: 0.47, recall: 0.10

Decision Tree and SVM have higher accuracy and precision but recall value is very low (under 0.3).So I have used Gaussian Naive Bayes as its performance is accepted for the final project, both precision and recall exceed 0.3.


## Parameters Tuning

I have used GridSearchCV to tune parameters of Decision Tree with both of min_samples_split and criterion parameters and the best estimators were 100 and gini respectively with SVM I have tried tuning gamma and C manually leaving kernel rbf.

The best parameters tuning here for SVM was (C=1000.0, gamma=0.01).

## Analysis Validation

I have validated my analysis using test_classifier function in tester.py which deployed cross validation within an iterative context to ensure that each data point got the chance in testing process so that validation results generalize well along dataset and don’t depend on any specific pattern or characteristics of the dataset.it is also a good solution for overfitting problem which may occur.

In our problem we have unbalanced small dataset only 18 POIs and 130 non-POI. we want to preserve the ratio of POI and NON-POI So stratified shuffle split is used here by which samples are first shuffled and then split into train and test sets returning stratified splits preserving the same percentage for each target class [POI and NON-POI] as in the complete set.

## Evaluation Metrics

I have used 2 evaluation metrics precision and recall to get an overall view of the algorithm's performance.recall was 0.32 meaning that the probability of the algorithm to correctly classify a person as POI provided that the person actually is POI is 0.32. precision was 0.43 which is the probability of the person being an actual POI if it is classified as a POI.
