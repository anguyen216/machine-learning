# Decision Tree
- Decision tree for binary classification
- The tree only does binary split
- This implementation does not weight each training example.  The implementation that includes examples' weight is included in AdaBoost folder

## Data
- dataset name: Default of credit card client
- source: [Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

## In this folder
- Implementation of decision tree as a tree structure using Python class object
- information_gain.py: helper codes that calculate the information gain of each split and determine the label of a split
- main.py: parse the given dataset (credit card default) and test the code using the data
- default of credit card client: large dataset for testing
- test.xlsx: small dataset that I created for testing.  Can be perfectly classified after 2 splits on age
