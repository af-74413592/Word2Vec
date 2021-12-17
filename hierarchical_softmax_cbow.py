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

    word_count = {}

    for sentence in datas:
        words = sentence.split(" ")
        for w in words:
            if w in word_count:
                word_count[w] = word_count[w] + 1
            else:
                word_count[w] = 1

    word_count = sorted(word_count.items(), key=lambda items: items[1])

    return datas,word_count

def sigmoid(x):
    return 1/(1+np.exp(-x))

class Node():
    def __init__(self,value,name=None):
        self._value = value
        self._name = name
        self._left=None
        self._right=None
        self._father=None
        self._pathnum = []
        self._code = None
        self._index = None

    def get_pathnum(self):
        path = self
        while path._father._index:
            self._pathnum.append(path._father._index)
            path = path._father
        return self._pathnum

class HuffmanTree():
    def __init__(self,char_weights):
        self.leafs = [Node(w,k) for k,w in char_weights]
        while len(self.leafs)!=1:
            self.leafs.sort(key=lambda node: node._value)
            father = Node(value=(self.leafs[0]._value+self.leafs[1]._value))
            father._left = self.leafs.pop(0)
            father._right = self.leafs.pop(0)
            father._left._father = father
            father._right._father = father
            self.leafs.append(father)
        self.root=self.leafs[-1]
        self.code = np.zeros(self.root.__sizeof__(), dtype=np.int)  # self.code用于保存每个叶子节点的哈夫曼编码

    def set_code(self, tree, length, code):
        node = tree
        s = ""
        if not node:
            return
        for i in range(length):
            s = s + str(self.code[i])
        if s:
            node._code = s
            node._index = 2**(len(s)) + int(s,base=2)
        if node._name:
            vec = np.zeros((length,1))
            for i in range(length):
                vec[i] = self.code[i]
            code[node._name] = (vec,node.get_pathnum(),node._index)
            return

        self.code[length] = 0
        self.set_code(node._left, length + 1, code)  # 递归左子树
        self.code[length] = 1
        self.set_code(node._right, length + 1, code)  # 递归右子树


    def get_code(self):
        code = {}
        self.set_code(self.root, 0, code)
        return code


if __name__ == '__main__':

    #propress()

    all_datas, word_count = get_data()

    tree = HuffmanTree(word_count)

    huffman_code_dict = tree.get_code()

    huffman_code_list = sorted(huffman_code_dict.items(),key=lambda items:items[1][2],reverse=True)

    max_len = huffman_code_list[0][1][2]+1

    embedding_num = 128

    epoch = 5

    lr = 0.001

    n_gram = 3

    w1 = wpath = np.random.normal(0,0.1,size=(max_len,embedding_num))


    for e in range(epoch):
        for sentence in tqdm(all_datas):
            words = sentence.split(" ")
            for now_idx, now_word in enumerate(words):
                other_words = words[max(0, now_idx - n_gram): now_idx] + words[now_idx + 1: now_idx + n_gram + 1]
                for other_word in other_words:
                    word_huffman = huffman_code_dict[now_word][0]
                    word_pathnum = huffman_code_dict[now_word][1]
                    word_index = huffman_code_dict[other_word][2]
                    for i,path in enumerate(reversed(word_pathnum)):
                        Gpath = (1 - word_huffman[i+1].item() - sigmoid(w1[word_index].T @ wpath[path])) * w1[word_index]
                        Gw1 = (1 - word_huffman[i+1].item() - sigmoid(w1[word_index].T @ wpath[path])) * wpath[path]
                        wpath[path] += lr * Gpath
                        w1[word_index] += lr * Gw1

    with open("vec.pkl1","wb") as f:
        pickle.dump([w1,huffman_code_dict,huffman_code_list],f)

    while True:
        try:
            word = input("请输入: ")
            v_w1 = w1[huffman_code_dict[word][2]]
            word_sim = {}
            for i in range(len(huffman_code_list)):
                v_w2 = w1[huffman_code_list[i][1][2]]
                theta_sum = np.dot(v_w1, v_w2)
                theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
                theta = theta_sum / theta_den
                word = huffman_code_list[i][0]
                word_sim[word] = theta
            words_sorted = sorted(list(word_sim.items()), key=lambda kv: kv[1], reverse=True)
            for word, sim in words_sorted[:20]:
                print(word, sim)
        except:
            print("没有该词! ")