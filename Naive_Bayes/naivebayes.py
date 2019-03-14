#!/usr/bin/env python3
from nltk.corpus import stopwords 
from math import log
import glob
import numpy as np
import random
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

class NaiveBayes():
    '''
    Implement Naive Bayes algorithm without using packages
    Specifically for sentiment analysis.  Predicting polarity of review / text
    '''

    def __init__(self):
        '''
        labels_count: dict{label : times appear in the dataset}
        label_words_count: dict{label: {word: times appear in the dataset}}
        punct: list(common punctuations)
        sw: list(english stopwords)
        '''

        self.labels_count = dict()
        self.label_words_count = dict()
        self.punct = [".", "?", ":", ";", ",", ")", "(", "/", "\"", "!"]
        self.sw = stopwords.words('english')


    def tokenize(self, review, remove_punct=False):
        '''
        splits individual review on space into words and outputs dictionary of 
        words and their count
        output dictionary does not include english stopwords

        -- parameters --
        review: string(individual review)
        remove_punct: boolean, False by default.  if True, remove punctuation in 
        review
        '''

        if remove_punct:
            s = review.strip()
            for p in self.punct:
                s = s.replace(p, '')
            tokens = s.lower().split(' ')
            words_count = {word: 1 for word in tokens}
            return words_count

        tokens = review.strip().lower().split(' ')
        words_count = {word: 1 for word in tokens}
        return words_count


    def update_counts(self, words_count, label):
        '''
        updates labels_count and label_words_count dictionaries

        -- parameters --
        words_count: dict{word : 1}
        label: string(label of review)
        '''
    
        # update label_words_count
        for word in words_count.keys():
            # add count to labels_count[label]
            if word in self.label_words_count[label]:
                self.label_words_count[label][word] += 1
            else:
                # smoothing
                self.label_words_count[label][word] = 2

            # add count to label_words_count[non_label][word]
            # smoothing
            for nl in self.label_words_count.keys():
                if nl != label and word not in self.label_words_count[nl]:
                    self.label_words_count[nl][word] = 1

        #update labels_count
        self.labels_count[label] += 1


    def log_likelihood(self, review):
        '''
        compute the likelihood of each label
        outputs the most likely label
        log(P(label|data)) = log(P(w1|label)) +..+ log(P(w_n|label)) + log(P(label))

        -- parameter --
        review: string(individual review)

        return
        best_label: string(most likely label)
        label_likelihood: float(value proportional to log(P(label|data)))
        '''

        # Set up
        best_label = ''; label_likelihood = -1000000000
        words_count = self.tokenize(review, remove_punct=True)
        total_obs = sum(self.labels_count.values())
    
        for label, count in self.labels_count.items():
            # likelihood = log(P(label | data))
            likelihood = log(count / total_obs)  # P(label)
            for word in words_count.keys():
                # add log(P(word|label))
                if word in self.label_words_count[label]:
                    likelihood += log(self.label_words_count[label][word] / count)
                else: 
                    # take care of the case when some word may have not been seen
                    likelihood += log(1 / count)

            if likelihood > label_likelihood: 
                label_likelihood = likelihood
                best_label = label

        return best_label

    
    def train(self, data, labels):
        '''
        Train the given data

        -- parameters --
        data: list(reviews)
        labels: list(labels correspond to reviews)
        '''
        
        # initiates labels_count and label_words_count dictionaries
        # allows continue training/updating with existing Naive Bayes model
        for label in set(labels):
            if label not in self.labels_count:
                self.labels_count[label] = 1
            if label not in self.label_words_count:
                self.label_words_count[label] = dict()

        # use each review in the dataset to update counts
        for idx, review in enumerate(data):
            words_count = self.tokenize(review, remove_punct=True)
            self.update_counts(words_count, labels[idx]) 


    def evaluate_accuracy(self, test_data, target_label):
        '''
        Computes the accuracy of Naive Bayes classifer

        -- parameters --
        test_data: list(reviews)
        target_label: list(true labels)

        return 
        correct: int(count of correct prediction)
        the ratio between the correct prediction and the total number of test data
        '''

        correct = 0
        for idx, review in enumerate(test_data):
            predict = self.log_likelihood(review)
            if predict == target_label[idx]: correct += 1

        return correct / len(target_label)


def input_file(path):

    '''
    takes the training data file and read it line by line
    return the text in each file as a string
    '''

    data = ''
    with open(path, 'r') as file:
        for line in file:
            data += line
    return data


def main():

    #===== Extracting training and testing data =====
    # collect all review files for both labels
    pos_files = glob.glob('aclImdb/train/pos/*.txt')
    neg_files = glob.glob('aclImdb/train/neg/*.txt')
    pos_test = glob.glob('aclImdb/test/pos/*.txt')
    neg_test = glob.glob('aclImdb/test/neg/*.txt')

    train_files = pos_files + neg_files
    test_files = pos_test + neg_test

    # get the label for each review
    # numerical labels; 1 for positive and 0 for negative
    train_labels = [1] * len(pos_files) + [0] * len(neg_files)
    test_labels = [1] * len(pos_test) + [0] * len(neg_test)

    # get the reviews and store them in training and testing data
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
    # CountVectorizer does text processing, punctuation and stopwords filtering
    # and transforms document into feature vector
    vect = CountVectorizer()
    
    # X_train: binary value indicate if a word appears in a review
    # X_train[i, j]: indication if word j is in review i
    X_train = vect.fit_transform(train_data)
    X_train[X_train > 1] = 1

    # transform testing data
    X_test = vect.transform(test_data)
    X_test[X_test > 1] = 1

    # instantiate NaiveBayes classfifier and train the classifier
    clf = MultinomialNB()
    clf.fit(X_train, train_labels)

    # accuracy evaluation
    predict = clf.predict(X_test)
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

    #===== Extracting words to answer questions =====
    words = nb.label_words_count
    labels_count = nb.labels_count
    sw = nb.sw
    pos_count = labels_count[1]
    neg_count = labels_count[0]

    # get the top 10 words (highest P(word|label))
    print()
    print("----- Top 10 words (highest P(word|label)) for each label -----")
    print("positive review")
    print(sorted(words[1], key=words[1].get, reverse=True)[:10])
    print("negative review")
    print(sorted(words[0], key=words[0].get, reverse=True)[:10])

    
    #===== probability of "fantastic" and "boring" =====
    print()
    print("----- probability of 'fantastic' and 'boring' for each label -----")
    print("positive review")
    print("probability of 'fantastic': ", words[1]['fantastic'] / pos_count)
    print("probability of 'boring': ", words[1]['boring'] / pos_count)
    print("negative review")
    print("probability of 'fantastic': ", words[0]['fantastic'] / neg_count)
    print("probability of 'boring'", words[0]['boring'] / neg_count)


    #===== caculate word polarity (attempt 1)=====
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
    with open('write_up/word_polarity.csv', 'w') as f:
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
