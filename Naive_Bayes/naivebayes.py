#!/usr/bin/env python3
from nltk.corpus import stopwords 
from math import log


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

