from typing import List

from collections import Counter

def simple_tfidf(word_t: str, doc: List[str], corpus: List[List[str]])-> float:

    len_corpus = len(corpus)


    def tf() -> float:
        return Counter(doc).get(word_t, 0) / max(Counter(doc_i).get(word_t, 0) for doc_i in corpus)
    def idf() -> float:
        return len_corpus / (sum(1 if word_t in doc_i else 0 for doc_i in corpus) + 1)

    return tf() * idf()