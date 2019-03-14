from decisiontree import *
import random


def read_data(path, multiheader=0, drop_cols=None):
    '''
    user pandas to read in excel files

    -- parameters --
    path: string(path to file)
    multiheader: int or int-list of row(s) to use as column headers
    drop_cols: int or int-list of column(s) to drop

    return the data in dataframe
    '''

    if multiheader:
        df = pd.read_excel(path, header=multiheader)
    else: df = pd.read_excel(path)

    if drop_cols != None:
        df.drop(df.columns[drop_cols], axis=1, inplace=True)

    return df


def main():
    filename = 'default of credit card clients.xls'
    tree = DecisionTree()
    data = read_data(filename, multiheader=1, drop_cols=0)
#    train = data.sample(frac=0.8, random_state=66)
#    test = data.drop(train.index)
    train_acc = tree.create_tree(data, 3)
#    test_acc = tree.compute_accuracy(test)

    tree.breadth_first_print()
    print('training accuracy ', train_acc)
#    print('testing accuracy ', test_acc)

if __name__ == '__main__':
    main()
