import numpy as np


def seq2onehot_346(x):
    """return one sequence"""
    maxlen = 346
    dim = 20
    aa2num = []
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    aa2int = dict((ami_acid, id_x+1) for id_x, ami_acid in enumerate(amino_acids))
    for aa in x:
        aa2num.append(aa2int[aa])

    """method2 more fast"""
    zero_matrix = np.eye(dim, dtype=np.int8)[np.array(aa2num) - 1]
    zero_matrix = np.pad(zero_matrix, ((0, maxlen - len(aa2num)), (0, 0)), 'constant')
    return zero_matrix


def seq2onehot_1000(x):
    """return one sequence"""
    maxlen = 1000
    dim = 20
    aa2num = []
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    aa2int = dict((ami_acid, id_x+1) for id_x, ami_acid in enumerate(amino_acids))
    for aa in x:
        aa2num.append(aa2int[aa])

    """methods2 more fast"""
    zero_matrix = np.eye(dim, dtype=np.int8)[np.array(aa2num) - 1]
    #print(len(x))
    zero_matrix = np.pad(zero_matrix, ((0, maxlen - len(aa2num)), (0, 0)), 'constant')
    return zero_matrix


def onehot_encode_346(datas):
    """if datas only has one sequence, please use seq2onehot function"""
    x = map(seq2onehot_346, datas)
    x = np.array(list(x))
    return x


def onehot_encode_1000(datas):
    """if datas only has one sequence, please use seq2onehot function"""
    x = map(seq2onehot_1000, datas)
    x = np.array(list(x))
    return x
