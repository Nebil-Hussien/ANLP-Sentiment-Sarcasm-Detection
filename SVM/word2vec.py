# simple Word2Vec utility class I (Kemal Afzal) wrote a while ago

import nltk
import pandas as pd
import numpy as np

class Word2Vec:
    dimensions = 300

    def __init__(self, seed):
        file = nltk.data._open("nltk:models/word2vec_sample/pruned.word2vec.txt")
        data = pd.read_csv(file,
            delimiter = " ",
            header = 0, names = ["word"] + [f"v{i}" for i in range(0, self.dimensions)],
            dtype = {"word": str},
            keep_default_na = False)

        self.__word_to_idx = dict[str, int]()

        for i in range(len(data)):
            self.__word_to_idx[data["word"][i]] = i

        vectors = [data[f"v{i}"].to_numpy(np.float32).reshape((len(data), 1)) for i in range(self.dimensions)]
        self.__mat = np.concatenate(vectors, axis = 1)

        self.__axis_std = np.std(self.__mat, axis = 0)

        self.__rand_state = np.random.RandomState(seed)

    def has_embedding(self, word):
        return word in self.__word_to_idx

    def get_word_embedding(self, word):
        return self.__mat[self.__word_to_idx[word]]

    def get_word_embedding_or_random(self, word):
        if self.has_embedding(word):
            return self.get_word_embedding(word)
        else:
            return self.__rand_state.random(self.dimensions) * self.__axis_std * 2 - self.__axis_std

    def sentence_dist(self, sent0_tokens, sent1_tokens) -> float:
        # cos(180) = -1
        accum = 0.0
        n = 0.0
        for word0 in sent0_tokens:
            if not self.has_embedding(word0):
                continue

            word0_vec = self.get_word_embedding(word0)

            for word1 in sent1_tokens:
                if not self.has_embedding(word1):
                    continue

                word1_vec = self.get_word_embedding(word1)

                accum += np.matmul(word0_vec, word1_vec.T)
                n += 1.0

        if n == 0.0:
            return -1.0
        else:
            return accum / n

if __name__ == "__main__":
    nltk.download("word2vec")
    w2v = Word2Vec()

    print(w2v.sentence_dist(["hello", "how", "are", "you"], ["hello", "how", "are", "you"]))
    print(w2v.sentence_dist(["hello", "how", "are", "you"], ["bye", "what", "is", "I"]))
    print(w2v.sentence_dist(["duck", "duck", "duck", "duck"], ["entropy", "chaos", "hope", "quaternion"]))
    print(w2v.sentence_dist(["cat"], ["bat"]))
    print(w2v.sentence_dist(["I"], ["you"]))
    print(w2v.sentence_dist(["this"], ["that"]))
    print(w2v.sentence_dist(["duck"], ["duck"]))
    print(w2v.sentence_dist(["now"], ["currently"]))
