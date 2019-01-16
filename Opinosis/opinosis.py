import spacy
import numpy as np
from spacy import displacy

nlp = spacy.load('en_core_web_sm')

sentences = "Hello, My name is Justin. This is another sentence. This is a third sentence. How are you Justin? I'm fine and you? Good. I love sentences!"

VSN_THRESHOLD = 3
REDUNDANCY_GAP = 4
MINIMUM_REDUNDANCY = 0

class Node:
    def __init__(self, word_unit):
        self.word = word_unit[:word_unit.index(":")]
        self.POS = word_unit[word_unit.index(":")+1:]
        self.word_unit = word_unit
        self.PRI = []
        self.edge = {}

    def getWordUnit(self):
        return self.word_unit

    def addPRI(self,SID,PID):
        self.PRI.append((SID,PID))
        return True
    def addEdge(self, next_node, PRI):
        if next_node not in self.edge:
            self.edge[next_node] = []
        self.edge[next_node] = PRI

def opinosisGraph(sentences):
    nodes_dict = {}
    sentence_list = []
    for x in sentences:
        if x in ".?!":
            sentence_list.append(sentences[0:sentences.index(x)+1])
            sentences = sentences[sentences.index(x)+2:]
    sentence_list = list(enumerate(sentence_list))
    for SID, sentence in sentence_list:
        PID = 0
        prev_word_unit = ""
        for word in sentence:
            word_unit = word + ":" + nlp(word)[0].pos_
            if word_unit not in nodes_dict:
                nodes_dict[word_unit] = Node(word_unit)
            nodes_dict[word_unit].addPRI(SID, PID)
            if prev_word_unit:
                nodes_dict[prev_word_unit].addEdge(nodes_dict[word_unit], (SID,PID))
            PID += 1
            prev_word_unit = word_unit
    return nodes_dict

def isVSN(node):
    return np.mean([x[0] for x in nodes_dict[node].PRI]) <= VSN_THRESHOLD

def intersect(PRI_overlap, node):
    pri_new = []
    for pri in PRI_overlap:
        last_sid, last_pid = pri[-1]
        for sid, pid in node:
            if sid == last_sid and pid - last_pid > 0 and pid - last_pid <= GAP:
                pri = pri[:]
                pri.append((sid, pid))
                pri_new.append(pri)
    return pri_new

def traverse(candidate_list, starting_node, score, PRI_overlap, sentence, len):
    if redundancy > MINIMUM_REDUNDANCY:
