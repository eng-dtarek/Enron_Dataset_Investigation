# Enron Dataset Investigation

This project is a part of the [Data Analyst Nanodegree](https://www.udacity.com/course/data-analyst-nanodegree--nd002) at [Udacity](https://www.udacity.com/)

-- Project Status: Completed

## Project Introduction
In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, there was a significant amount of typically confidential information entered into public record, including tens of thousands of emails and detailed financial data for top executives. In this project, I play detective, and put my skills to use by building a person of interest identifier based on financial and email data made public as a result of the Enron scandal.

## Methods Used

- Machine Learning
- Data Exploration
- Features Selection
- Analysis Validation
- Evaluation Metrics

## Technologies
- Python >= 2.7
- sklearn

## Project Description

The purpose of this project is to build a prediction model to identify POI from non POI in Enron dataset given some features using Machine learning algorithms to detect the underlying patterns for POI from the given features.
I explored the [dataset](https://github.com/eng-dtarek/Enron_Dataset_Investigation/blob/master/my_dataset.pkl) to get an overview and define the outliers, used SelectKBest for features selection, applied multiple Machine Learning Algorithms (Gaussian Naive Bayes, Decision Tree, SVM), validated my analysis and used evaluation metrics to measure their performance.
The best resulted model achieved 85% accuracy, 0.43 precision, and 0.32 recall using Gaussian Naive Bayes. (for more details see this [documentation](https://github.com/eng-dtarek/Enron_Dataset_Investigation/blob/master/Enron%20Submission%20Free-Response%20Questions.pdf))

## Getting Started

1. Clone this repo (for help see this [tutorial](https://help.github.com/en/articles/cloning-a-repository)).
2. Install the above technologies.
3. Run poi_id.py file to dump classifier, dataset, and features_list.
4. Run tester.py file for the test results.

## References

See this file [references](https://github.com/eng-dtarek/Enron_Dataset_Investigation/blob/master/references.txt)

