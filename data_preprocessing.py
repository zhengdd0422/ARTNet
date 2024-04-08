

def read_fasta(filename, label=1):
    """return ids, fasta, and labels, if fasta file has no labels, just ignore it"""
    f_open = open(filename, "r")
    ids = []
    seqs = []
    labels = []
    id_name = ""
    each_seq = ""
    for i, line in enumerate(f_open.readlines()):
        line = line.strip("\n")
        if line.find(">") >= 0:
            if i != 0:
                ids.append(id_name)
                seqs.append(each_seq)
                labels.append(label)
            id_name = line
            each_seq = ""
        else:
            if each_seq != "":
                each_seq += line
            else:
                each_seq = line

    # remember to save the last sequence
    ids.append(id_name)
    seqs.append(each_seq)
    labels.append(label)
    f_open.close()

    return ids, seqs, labels


def split_sequences_step(sequence, lens, step=1):
    x = list()
    for i in range(0, len(sequence), step):
        end_ix = i + lens

        if end_ix <= len(sequence):
            seq_ix = sequence[i:end_ix]
            x.append(seq_ix)
        else:
            seq_ix = sequence[i:len(sequence)]
            x.append(seq_ix)
            break
    return x


def read_id_info(filename, signal="&"):
    all_data = {}
    f_file = open(filename, "r")
    for eachline in f_file.readlines():
        if eachline.find("seq_id") < 0:
            eachline = eachline.strip()
            lists = eachline.split(signal)
            all_data[lists[0]] = lists[1] + signal + lists[2]

    return all_data
