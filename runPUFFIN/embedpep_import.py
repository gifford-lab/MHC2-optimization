import numpy as np

def embed(seq, mapper):
    return np.asarray([mapper[s] for s in seq]).transpose()

def lenpep_feature(pep):
    lenpep = len(pep) - pep.count('J')
    f1 = 1.0/(1.0 + np.exp((lenpep-args.expected_pep_len)/2.0))
    return f1, 1.0-f1

def embed_all(pep_f, mapper):
    pep = []
    peplen = []

    cnt = 0
    bs_cnt = 0
    for pep_line in pep_f:
        pep.append(embed(pep_line, mapper))
    return pep

#Takes as input a list of peptides (length at most 40)
def getPeps(peplist):
    with open("./onehot_first20BLOSUM50", 'rt') as fin:
        mapper = dict()
        for x in fin:
            line = x.split()
            mapper[line[0]] = [float(z) for z in x.split()[1:]]

    def addpadding(seq):
        if len(seq) > 9:
            1/0
        else:
            seq = seq + ('J' * (9-len(seq)))
        return seq
    
    peplist = [addpadding(seq) for seq in peplist]
    return embed_all(peplist, mapper)
