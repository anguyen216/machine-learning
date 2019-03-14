#!/usr/bin/env python3
from information_gain import *
import random


class Node:
    '''
    Base of the decision tree
    holds the information at each split
    '''
    
    def __init__(self):
        '''
        -- attributes --
        self.feature: feature to split
        self.value: value to split 
        self.left: node that has data[self.feature] <= self.value
        self.right: node that has data[self.feature] > self.value
        self.entropy: entropy of the split
        self.label: the label of the node (non-leaf nodes labels are the majority
        of data being passed into the node)
        self.gain: the information gain of split on this feature
        self.depth: the level the node is at
        '''

        self.feature = None
        self.value = None
        self.left = None
        self.right = None
        self.label = None
        self.gain = None
        self.depth = 0

    def get_children(self):
        return [self.left, self.right]

    def __repr__(self):
        return '%s <= %s, class = %s, gain = %s, depth = %s' % (self.feature, self.value, self.label, self.gain, self.depth)


class DecisionTree:
    '''
    Binary classification
    Binary tree
    '''

    def __init__(self):
        self.root = None
        self.num_levels = 0
        self.accurate_count = 0


    def determine_split(self, data):
        '''
        -- parameter --
        data: pandas dataframe of training data
        return
        1) the feature to split on
        2) the value to split on
        3) the entropy of that split
        4) the prediction for each children
        '''
        #-- setup --
        max_gain = -1 
        best_feature = None
        best_val = None

        # assume last column is label column
        # exclude label column
        features = data.columns.values[:-1] 

        for ft in features:
            gain, value = information_gain(data, ft)
            if gain > max_gain:
                max_gain = gain
                best_feature = ft
                best_val = value

        left_label, right_label = majority_label(data, best_feature, best_val)
        return max_gain, best_feature, best_val, left_label, right_label


    def do_split(self, data, depth=0):
        '''
        -- parameters --
        data: dataframe of examples to consider
        depth: int(current depth of the node)

        builds the decision tree using determine_split outputs
        keeps track of number of accurate classification
        returns leaf nodes
        '''
        #-- BASE CASES --
        # BC 1:
        # if all example are in one class, 
        # return leaf node with that label
        label = data.columns.values[-1]
        if sum(data[label]) == len(data): 
            leaf = Node()
            leaf.label = 1
            leaf.depth = depth
            self.accurate_count += len(data)
            return leaf
        if sum(data[label]) == 0: 
            leaf = Node()
            leaf.label = 0
            leaf.depth = depth
            self.accurate_count += len(data)
            return leaf
        
        # BC 2:
        # reach max depth
        if depth == self.num_levels:
            leaf = Node()
            leaf.label = find_majority(data)
            leaf.depth = depth
            self.accurate_count += len(data[data[label] == leaf.label])
            return leaf

        # create the split node
        root = Node()
        gain, feature, value, left_label, right_label = self.determine_split(data)
        root.gain = gain
        root.feature = feature
        root.value = value
        root.label = find_majority(data)
        root.depth = depth
        left_data = data[data[feature] <= value].copy()
        right_data = data[data[feature] > value].copy()

        # if the split leaves one side with no example 
        # create leaf node with label of the current data
        if len(left_data) == 0:
            assert(len(right_data) > 0)  # debug
            root.left = Node()
            root.left.label = root.label
            root.left.depth = depth
            depth += 1
            root.right = self.do_split(right_data, depth)
        elif len(right_data) == 0:
            assert(len(left_data) > 0)  # debug
            root.right = Node()
            root.right.label = root.label
            root.right.depth = depth
            depth += 1
            root.left = self.do_split(left_data, depth)
        else:
            assert(len(left_data) != 0 and len(right_data) != 0)  # debug
            depth += 1
            root.left = self.do_split(left_data, depth)
            root.right = self.do_split(right_data, depth)

        return root


    def test(self, example):
        '''
        traverse the tree to determine the classfication of an example
        -- parameter --
        example: pandas.series of a single example to be tested

        return the predicted label of the example
        '''
        current = self.root
        while current.feature is not None:     
            v = current.value
            if example[current.feature] <= v: current = current.left
            else: current = current.right
        return current.label


    def test_whole(self, current_node, test):
        '''
        recursively travel the trained model to test the entire dataset
        -- parameters --
        current_node: the current node being examined
        test: the set of examples being considered

        return: the new dataframe of all examples with prediction column
        '''
        # BC: reach leaf node, labels data using leaf's label
        if current_node.feature is None:
            res = test.copy()
            res['pred'] = current_node_label
            return res

        v = current_node.value
        feature = current_node.feature
        left = test[test[feature] <= v]
        right = test[test[feature] > v]
        left = self.test_whole(current_node.left, left)
        right = self.test_whole(current_node.right, right)
        return pd.concat([left, right])


    def breadth_first_print(self):
        '''
        print all nodes in the tree in breadth first order
        left child first
        '''
        queue = [self.root]
        while len(queue) > 0:
            node = queue.pop(0)
            print(node)
            children = node.get_children()
            if children[0] != None:
                queue.append(children[0])
            if children[1] != None:
                queue.append(children[1])


    def compute_accuracy(self, test_data):
        '''
        -- parameter --
        test_data: dataframe of examples to be tested using 
        the trained tree
        
        return accuracy on the test set
        '''
        label = test_data.columns.values[-1]
        new_df = self.test_whole(self.root, test_data)
        prediction = new_df['pred'].values
        truth = new_df[label].values
        correct_count = (prediction == truth).sum()
        return correct_count/len(test_data)


    def create_tree(self, data, num_levels):
        '''
        uses all the data in the path file to train the tree
        -- parameters --
        path: string(path to data file)
        num_levels: int(tree maximum depth, depth starts at 0)
        
        This function
        1) reads the data
        2) determines all of the splits and their labels
        3) saves the tree 
        4) outputs the accuracy of the tree on the training set
        '''
        self.num_levels = num_levels
        self.root = self.do_split(data)
        training_accuracy = self.accurate_count / len(data)
        return training_accuracy

