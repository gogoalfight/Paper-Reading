import jieba.posseg as pg

def tokenize(corpus):

    word_bags=[]
    for index in range(len(corpus)):
        result =[]
        text =corpus[index]
        words=pg.cut(text)
        for word,flag in words:
            result.append(word)

        word_bags.append(result)

    return word_bags