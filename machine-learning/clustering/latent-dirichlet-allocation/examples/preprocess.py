import gensim


def sent_to_words(sentences: list):
    for sent in sentences:
        yield(gensim.utils.simple_preprocess(str(sent), deacc=True))


if __name__ == "__main__":
    from load_data import load_new_group_dataset

    df = load_new_group_dataset()

    data = list(df["content"])

    data_words = list(sent_to_words(data))

    print(data_words[:1])