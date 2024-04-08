# -*- coding: utf-8 -*-
"""
@author : Dandan Zheng, 2024/04/03
 predict object sequence
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from ADPRT_models import *
from keras.models import Model
from encodes import *
from keras import backend as K
from data_preprocessing import *
import os
import logging
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--usefile_id", type=str, default='1', help="the path of the data sets")
parser.add_argument("--method_signal", type=str, default='1', help="the number of epochs", choices=['0', '1', '2'])                                                                                        
args = parser.parse_args()
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['PYTHONHASHSEED'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
np.random.seed(42)
session_conf = tf.compat.v1.ConfigProto(allow_soft_placement=True)
session_conf.gpu_options.allow_growth = True
session_conf.gpu_options.per_process_gpu_memory_fraction = 0.5
tf.set_random_seed(42)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
MAX_INT = np.iinfo(np.int32).max

# parameters
# dir
data_dir= os.getcwd()
model_dir = data_dir + "trained_models/"
tmp_path = data_dir + "tmp/"
f_open = tmp_path + args.usefile_id + ".fasta"
all_ids, all_seqs = read_fasta(f_open)[0], read_fasta(f_open)[1]

# models
# pos_art_346
model1 = adprtnet((346, 20))
model1.load_weights(model_dir + "pos_art_346.h5")
mmodel1_scoring = Model(inputs=model1.input, outputs=model1.output)
# pos_art_346_random
model2 = adprtnet((346, 20))
model2.load_weights(model_dir + "pos_art_346_random.h5")
mmodel2_scoring = Model(inputs=model2.input, outputs=model2.output)
# pos_whole
model3 = adprtnet((1000, 20))
model3.load_weights(model_dir + "pos_whole.h5")
mmodel3_scoring = Model(inputs=model3.input, outputs=model3.output)

len_thred = {
    "pos_art_346": 346,
    "pos_art_346_random": 346,
    "pos_whole": 1000
}

all_class_save = {}
all_class = []
for k, each_seq in enumerate(all_seqs):
    each_seq_class = {}
    if (each_seq.find('X') < 0) and (each_seq.find('B') < 0) and (each_seq.find('Z') < 0) and (
            each_seq.find('U')) < 0 and (each_seq.find('J') < 0) and (each_seq.find('*') < 0) and (each_seq.find('-') < 0):
        for each_mm in ["pos_art_346", "pos_art_346_random", "pos_whole"]:
            sub_seqs = []
            if len(each_seq) <= len_thred[each_mm]:
                sub_seqs.append(each_seq)
            else:
                sub_seqs = split_sequences_step(each_seq, len_thred[each_mm], step=1)
            if each_mm == "pos_art_346":
                seq_encode = onehot_encode_346(sub_seqs)
                scores = mmodel1_scoring.predict(seq_encode)
            elif each_mm == "pos_art_346_random":
                seq_encode = onehot_encode_346(sub_seqs)
                scores = mmodel2_scoring.predict(seq_encode)
            elif each_mm == "pos_whole":
                seq_encode = onehot_encode_1000(sub_seqs)
                scores = mmodel3_scoring.predict(seq_encode)

            scores_list = list(scores.reshape(len(scores)))
            max_score = max(scores_list)
            if max_score > 0.5:
                each_seq_class[each_mm] = 1
            else:
                each_seq_class[each_mm] = 0

        if args.method_signal == "0":
            if list(each_seq_class.values())[0] + list(each_seq_class.values())[1] + list(each_seq_class.values())[2] >=1:
                all_class.append(1)
            else:
                all_class.append(0)
        elif args.method_signal == "1":
            if list(each_seq_class.values())[0] + list(each_seq_class.values())[1] + list(each_seq_class.values())[2] >=2:
                all_class.append(1)
            else:
                all_class.append(0)
        elif args.method_signal == "2":
            if list(each_seq_class.values())[0] + list(each_seq_class.values())[1] + list(each_seq_class.values())[2] ==3:
                all_class.append(1)
            else:
                all_class.append(0)
        all_class_save[k] = str(each_seq_class["pos_art_346"]) + "&" + str(each_seq_class["pos_art_346_random"]) + "&" + str(each_seq_class["pos_whole"])
    else:
        all_class.append(0)
        all_class_save[k] = "-&-&-"

f_save = open(data_dir + "predict_out/" + args.usefile_id + "_predict.txt", "a")
f_save.write("seq_id&pos_art_346&pos_art_346_random&pos_whole\n")
if max(all_class) == 1:
    for jj, each_ids in enumerate(all_ids):
        if all_class[jj] == 1:
            f_save.write("{}&{}\n".format(each_ids, all_class_save[jj]))
else:
    f_save.write("all negative&-&-&-\n")


f_save.close()
