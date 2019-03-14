#!/usr/bin/env python3
from naivebayes import *
import glob
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import random
import numpy as np


def input_file(path):
    '''
    takes the moview training data files and read them line by line
    return the text in each file as a string
    '''

    data = ''
    with open(path, 'r') as file:
        for line in file:
            data += line

    return data


def main():
    #====== Extract training and testing data =====
    # collect all review files for both labels
    pos_files = glob.glob('aclImdb/train/pos/*.txt')
    neg_files = glob.glob('aclImdb/train/neg/*.txt')
    pos_test = glob.glob('aclImdb/test/pos/*.txt')
    neg_test = glob.glob('aclImdb/test/neg/*.txt')

    train_files = pos_files + neg_files
    test_files = pos_test + neg_test
    
    # creat the label for each review
    # numerical labels; 1 for positive and 0 for negative
    train_labels = [1] * len(pos_files) + [0] * len(neg_files)
    test_labels = [1] * len(pos_test) + [0] * len(neg_test)

    # extract data from files
    train_data = [input_file(filename) for filename in train_files]
    test_data = [input_file(filename) for filename in test_files]

    #===== test Naive Bayes class =====
    nb = NaiveBayes()
    nb.train(train_data, train_labels)
    ans = nb.evaluate_accuracy(test_data, test_labels)

    print("----- Self-implemented Naive Bayes -----")
    print("accuracy score (entire test set): %s" % (ans))

    #===== Scikit-learn Naive Bayes =====
    # Multinomial Naive Bayes
    # CountVectorizer process and transform data into feature vector
    # remove punctuation and stopwords
    vect = CountVectorizer()

    # X_train and X_test: binary values indicate appearance of words
    X_train = vect.fit_transform(train_data)
    X_train[X_train > 1] = 1
    X_test = vect.transform(test_data)
    X_test[X_test > 1] = 1

    clf = MultinomialNB()
    clf.fit(X_train, train_labels)

    predict = clf.predict(X_test)
    # to match NaiveBayes way to getting accuracy
    print()
    print("----- Scikit-learn Multinomial Naive Bayes -----")
    print('accuracy score (entire test set): ', \
              np.mean(predict == np.asarray(test_labels)))

    #===== test 2 models with 10 subsets =====
    print()
    print("----- compare two models with different test subsets -----")
    print("test set size | scratch model | scikit model")
    for i in range(10):
        sample_idx = random.sample(range(25000), k=random.randrange(1000, 20000, 1))
        sample_test = [test_data[i] for i in sample_idx]
        sample_labels = [test_labels[i] for i in sample_idx]

        test_subset = vect.transform(sample_test)
        test_subset[test_subset > 1] = 1
        scikit_predict = clf.predict(test_subset)
        scikit_score = np.mean(scikit_predict == np.asarray(sample_labels))

        scratch_score = nb.evaluate_accuracy(sample_test, sample_labels)
        print(len(sample_idx), scratch_score, scikit_score)

    #===== caculate word polarity =====
    words = nb.label_words_count
    labels_count = nb.labels_count
    sw = nb.sw
    pos_count = labels_count[1]
    neg_count = labels_count[0]
    w_dict = dict()
    for word in words[1].keys():
        if word not in sw:
            w = dict()
            count_pos = words[1][word]
            count_neg = words[0][word]
            count = count_pos + count_neg
            polarity = count_pos - count_neg
            w_dict[word] = polarity

    top_pos = sorted(w_dict, key=w_dict.get, reverse=True)[:10]
    top_neg = sorted(w_dict, key=w_dict.get, reverse=False)[:10]
    print()
    print("----- top 10 words for each label using word polarity -----")
    print("positive")
    print(top_pos)
    print("negative")
    print(top_neg)

    # write w_dict to csv
    header = ['word', 'count', 'polarity,group\n']
    with open('visualization/word_polarity_gh.csv', 'w') as f:
        f.write(','.join(header))
        for key in w_dict.keys():
            if (words[1][key] + words[0][key]) > 200:
                if key in top_pos:
                    f.write(key + ',' + str(words[1][key] +  words[0][key]) + ',' +\
                        str(w_dict[key]) + ',positive\n')
                elif key in top_neg:
                    f.write(key + ',' + str(words[1][key] + words[0][key]) + ',' +\
                            str(w_dict[key]) + ',negative\n')
                else:
                    f.write(key + ',' + str(words[1][key] + words[0][key]) + ',' +\
                            str(w_dict[key]) + ',neutral\n')

if __name__ == '__main__':
    main()
