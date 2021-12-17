import pandas as pd
import jieba
import numpy as np
import re
from tqdm import tqdm
import pickle

def propress(file = "数学原始数据.csv"):
    p = r"[!@#$%^&*()a-zA-Z|?\.,\-+！￥（），。、？【】；‘’：<>:'\"\\~`\[\]≠0-9/﹣=△{}“”]"

    datas = list(pd.read_csv(file,names=["content"],encoding="gbk")["content"])

    new_datas = []

    for sentence in datas:
        sentence = re.sub(p,"",sentence)
        if sentence:
            new_datas.append(" ".join(jieba.lcut(sentence)))

    with open("new_data.txt","w",encoding = "utf-8") as f:
        f.write("\n".join(new_datas))

def get_data(file = "new_data.txt"):
    with open(file,encoding='utf-8') as f:
        datas = f.read().split("\n")

    word_2_index = {}
    index_2_word = []
    full_table = []

    for sentence in datas:
        words = sentence.split(" ")
        for w in words:
            if w not in word_2_index:
                word_2_index[w] = len(word_2_index)
                index_2_word.append(w)
            full_table.append(word_2_index[w])

    corpus_len = len(word_2_index)

    return datas,word_2_index,index_2_word,full_table,corpus_len

def sigmoid(x):
    return 1/(1+np.exp(-x))

def make_samples(words,index):
    global word_2_index, corpus_len, neg_rate, full_table
    now_word_index = word_2_index[words[index]]
    other_words = words[max(0, index - n_gram): index] + words[index + 1: index + n_gram + 1]
    other_words_index = [word_2_index[i] for i in other_words]

    t = np.random.randint(0,len(full_table),size = (neg_rate*len(other_words_index)* 2))
    t = [i for i in t if i not in other_words_index + [now_word_index]][:neg_rate*len(other_words_index)]

    samples = []

    for i in other_words_index:
        samples.append((now_word_index,i,1))

    for i in t:
        samples.append((now_word_index,full_table[i],0))

    return samples

def pro_samples(samples):

    words_index = []
    label = []
    for sample in samples:
        words_index.append(sample[1])
        label.append(sample[2])
    return samples[0][0],words_index,np.array(label).reshape(1,-1)

def word_voc(word):
    return w1[word_2_index[word]]

def voc_sim(word, top_n):
    v_w1 = word_voc(word)
    word_sim = {}
    for i in range(len(word_2_index)):
        v_w2 = w1[i]
        theta_sum = np.dot(v_w1, v_w2)
        theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
        theta = theta_sum / theta_den
        word = index_2_word[i]
        word_sim[word] = theta
    words_sorted = sorted(list(word_sim.items()), key=lambda kv: kv[1], reverse=True)
    for word, sim in words_sorted[:top_n]:
        print(word, sim)

if __name__ == '__main__':

    #propress()

    all_datas, word_2_index, index_2_word, full_table, corpus_len = get_data()

    embedding_num = 128

    epoch = 5
    lr = 0.01
    n_gram = 6
    neg_rate = 10

    w1 = np.random.normal(0,1,size=(corpus_len,embedding_num))

    w2 = np.random.normal(0,1,size=(w1.T.shape))

    for e in range(epoch):
        for sentence in tqdm(all_datas):
            words = sentence.split(" ")
            for now_idx, now_word in enumerate(words):
                samples = make_samples(words,now_idx)
                now_word_idx, word_idx, label = pro_samples(samples)
                hidden = 1 * w1[now_word_idx,None]

                pre = hidden @ w2[:, word_idx]

                pro = sigmoid(pre)

                #loss = -np.sum(label * np.log(pro) + (1-label) * np.log(1-pro))

                G2 = pro - label

                delta_w2 = hidden.T @ G2

                G1 = G2 @ w2[:, word_idx].T

                delta_w1 = G1

                w1[None,now_word_idx] -= lr * delta_w1
                w2[:,word_idx] -= lr * delta_w2 / len(label)

    with open("vec.pkl2","wb") as f:
        pickle.dump([w1,w2,word_2_index,index_2_word],f)

    while True:
        try:
            word = input("请输入: ")
            voc_sim(word, 20)
        except:
            print("没有该词! ")
