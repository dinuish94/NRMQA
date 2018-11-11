from __future__ import print_function
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json, load_model
from functools import reduce
import numpy as np
import re
import os
from keras import backend as K



def tokenize(sent):
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
 
    data = []
    story = []
    for line in lines:
        line = line.strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data

def parse_input_story(input_story,question):
    return [(tokenize(input_story),tokenize(question))]
    
    

def get_stories(f, only_supporting=False, max_length=None):

    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    X = []
    Xq = []
    for story, query in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        X.append(x)
        Xq.append(xq)
    return (pad_sequences(X, maxlen=story_maxlen),
            pad_sequences(Xq, maxlen=query_maxlen))

datadir = "data/tasks_1-20_v1-2/en-10k/"

def load_babi_task(taskid):

    K.clear_session()
    
    print("Task ID : " ,taskid)
    filenameformat = "qa{}_".format(taskid)
    for file in os.listdir(datadir):
        if taskid == -1 or filenameformat in file and 'train' in file:
            train_file = file
    
        elif taskid == -1 or filenameformat in file and 'test' in file:
            test_file = file

    print("Train File", train_file)
    print("Test File", test_file)

    global train_stories
    train_stories = get_stories(open(datadir+train_file,'r'))

    global test_stories
    test_stories = get_stories(open(datadir+test_file,'r'))

    global loaded_model
    loaded_model = load_model('task%s.h5'%(taskid))
    loaded_model.load_weights('task%s.h5'%(taskid))
    print("Loaded model from disk")

    global vocab
    vocab = set()
    for story, q, answer in train_stories + test_stories:
        vocab |= set(story + q + [answer])

    
    vocab = sorted(vocab)

    global vocab_size
    vocab_size = len(vocab) + 1

    global story_maxlen
    story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))

    global query_maxlen
    query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

    global word_idx
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

def get_vocab():
    return list(vocab)

def get_ran_task():
#    n = np.random.randint(0,1000)
    n = 2
    story_list = test_stories[n][0]
    story =' '.join(word for word in story_list)
    question_list = test_stories[n][1]
    ques =' '.join(word for word in question_list)
    ans = test_stories[n][2]
    return [story,ques,ans]

def getAnswer(parsed_stories):
    inputs_train, queries_train = vectorize_stories(parsed_stories,word_idx,story_maxlen,query_maxlen)
    pred_results = loaded_model.predict(([inputs_train, queries_train]))
    val_max = np.argmax(pred_results[0])
    for key, val in word_idx.items():
        if val == val_max:
            k = key
    return k