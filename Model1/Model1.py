import codecs
from collections import defaultdict
from nltk import word_tokenize

class IBM_1:

    def __init__(self,lang1,lang2):
        self.lang1_file = lang1
        self.lang2_file = lang2
        self.corpus_file = self.build_corpus()
        self.translation_prob = self.init_trans_prob()


    def build_corpus(self):
        files = []
        files.insert(0,(codecs.open(self.lang1_file,"r","utf-8")))
        files.insert(1,(codecs.open(self.lang2_file,"r","utf-8")))   
        corpus_file = dict()
        i = 0
        while i<10:
            sentences1 = tuple(word_tokenize("NULL "+files[0].readline().strip("\n").lower()))
            sentences2 = tuple(word_tokenize("NULL "+files[1].readline().strip("\n").lower()))
            i += 1
            corpus_file[sentences1] = sentences2
        return corpus_file

    def init_trans_prob(self):
        num_l2 = len(set(word for (lang1,lang2) in self.corpus_file.items() for word in lang2))
        translation_prob = defaultdict(lambda: float(1/num_l2))
        print(translation_prob.items())
        return translation_prob

    def trainer_module(self,epochs=1):

        for epoch in range(epochs):
            l1_given_l2 = defaultdict(float)
            net = defaultdict(float)
            sentence_level = defaultdict(float)

            for (sent_lang1,sent_lang2) in self.corpus_file.items():
                for word in sent_lang1:
                    for word2 in sent_lang2:
                        sentence_level[word] += self.translation_prob[(word,word2)]
                for word in sent_lang1:
                    for word2 in sent_lang2:
                        l1_given_l2[(word,word2)] += (self.translation_prob[(word,word2)]/sentence_level[word])
                        net[word2] += (self.translation_prob[(word,word2)]/sentence_level[word])
            for (word,word2) in l1_given_l2:
                self.translation_prob[(word,word2)] = l1_given_l2[(word,word2)]/net[word2]
        print(self.translation_prob)
        return self.translation_prob

lang1 = 'data/en'
lang2 = 'data/fr'
obj1 = IBM_1(lang1,lang2)
obj1.trainer_module()
