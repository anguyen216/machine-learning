# machine learning
##### last updated: 3/14/19

This folder includes several common machine learning algorithms.

The purpose of this repository is to help with the understanding of the mathematical foundation (and how to implement them) behind common machine learning algorithms.  As a result, these implementations avoid using advance python libraries and packages, which if used in practice can significantly speed up the training process and be more general (for example, the naive bayes implementation in this folder, though can classify more than 2 classes, only works with text data).  In addition, for the sake of avoid reinventing the wheel as well as avoid excessive looping through the data (making the code clunky), I used pandas and numpy to store and filter the data and to do some simple calculations

Each folder includes at least the following:
1. the implementation of each algorithm (often as a class), 
2. some data to test the code, and 
3. a main file that parses the testing data and tests the code

To run main files, simply do 
```
python main.py
```

### To be included
List of files/algorithms to be included

*Very fine print: I am a senior so I don't have much time, but I will try to check all of these out as soon as I can :)*
- [x] Testing data for each algorithm
- [x] Naive Bayes
- [x] Decision Tree
- [ ] AdaBoost using Decision Stump
- [ ] Single perceptron **(huge maybe though, as this assignment may be used again in the future and I don't want to spoil the fun for anyone wanting to struggle with it a bit)**
- [ ] README files for each algorithm
- [ ] algorithm evaluation functions

### Note
- All codes follow python3 syntax
- Some files may have excessive comment, this is because these files are from my class projects (I have edit them so they are easier to read, as well)
- Everyone is welcome to check these implementations out and try them.  I do appreciate feedback (if you see something wrong, or some changes that can make the code **faster**, please do let me know)
- I am aware that some python libraries and packages can make some of my implementation much faster.  However, as the nature of my class assignment was to help students understand the algorithm in the most apparent way, I refrained from using advanced packages (such as scikit learn).  I only used those for result comparison
