#!/usr/bin/env python3
import pandas as pd
import numpy as np


def log2(x):
    '''
    Calculate log base 2 of a number
    log2(0) = 0 (by default)
    '''
    if x == 0: return 0
    return np.log2(x)


def find_midpoints(arr):
    '''
    given a numpy array of values
    returns a sorted numpy array of unique values of midpoints
    '''

    values = np.unique(arr)
    if len(values) == 1: return values
    res = np.array([(values[i] + values[i+1])/2 for i in range(len(values) - 1)])
    return res


def bool_entropy(q):
    '''
    formula from Russel and Norvig's book pg.708
    given the probability of a Boolean random variable
    return the random variable's entropy
    '''

    return -(q * log2(q) + (1 - q) * log2(1 - q))


def bool_entropy_vector(q):
    '''
    given a vector of probabilities qs of Boolean random variables
    return a vector of entropy of each value
    '''
    # fill 0s with 1s to avoid np.log2(0) == undefined
    # force np.log2(0) = 0
    q[q == 0] = 1
    p = 1 - q  # the complement of q
    p[p == 0] = 1
    return -(q * np.log2(q) + p * np.log2(p))


def information_gain(data, ft):
    '''
    -- parameters --
    data: dataframe
    ft: feature to split the data on

    calculate the information gain of all possible split points
    return best information gain and split value
    '''

    #-- setup --
    # assume the label column is the last column
    # get the positive label portion of the data
    n = len(data)
    label = data.columns.values[-1]
    pos = data.loc[data[label] == 1] 
    m_pos = len(pos)

    # calculate the total entropy 
    total_entropy = bool_entropy(m_pos/n)

    # no split
    if len(np.unique(data[ft].values)) == 1:
        return 0, data[ft].values[0]

    # find all possible split points
    mp = find_midpoints(data[ft].values)
    
    # calculate entropy of split
    n_left = np.array([(data[ft].values <= x).sum() for x in mp])
    n_right = n - n_left
    left_pos = np.array([(pos[ft] <=x).sum() for x in mp])
    right_pos = m_pos - left_pos
    h_left = bool_entropy_vector(left_pos/n_left)
    h_right = bool_entropy_vector(right_pos/n_right)

    # information gains of all split points
    gain = total_entropy - (n_left/n) * h_left - (n_right/n) * h_right
    max_idx = np.argmax(gain)
    return gain[max_idx], mp[max_idx]


def find_majority(data):
    '''
    -- parameter --
    data: data frame
    
    return the majority label in the dataframe
    if it's 50/50 return 0 always
    '''
    label = data.columns.values[-1]
    return 1 if sum(data[label]) > (len(data)/2) else 0


def majority_label(data, feature, val):
    '''
    -- parameters --
    data: dataframe
    feature: feature to split the data on
    val: value to split the data on

    return the classification for each leaf
    '''
    label = data.columns.values[-1]
    left = data[data[feature] <= val]
    right = data[data[feature] > val]
    left_label = find_majority(left)
    right_label = find_majority(right)
    return left_label, right_label
