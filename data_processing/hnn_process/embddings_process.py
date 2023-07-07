"""
从大词典中获取指定语料的词典
将数据处理成待打标签的形式
"""

import pickle
from typing import List
import numpy as np
from gensim.models import KeyedVectors


def load_word_vectors(type_vec_path: str, word_dict: dict, pre_word_vec: List[float]) -> List[float]:
    """
    加载词向量矩阵
    :param type_vec_path: 二进制词向量文件路径
    :param word_dict: 词汇字典
    :param pre_word_vec: 原有词向量
    :return: 更新后的词向量矩阵
    """
    model = KeyedVectors.load(type_vec_path, mmap='r')
    word_dict_list = list(word_dict.keys())
    word_vectors_list = pre_word_vec.tolist()
    fail_word_list = []

    # 为无法找到的词生成随机嵌入
    rng = np.random.RandomState(None)

    unk_embedding = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()

    # 查找词向量，找到的词追加到列表中，找不到的词添加到失败词列表
    for word in word_dict_list:
        try:
            word_vectors_list.append(model.wv[word])
        except:
            fail_word_list.append(word)
            word_vectors_list.append(unk_embedding)

    # 返回更新后的词向量列表和失败词列表
    return word_vectors_list, fail_word_list


def build_new_dict(type_vec_path: str, type_word_path: str, final_vec_path: str, final_word_path: str) -> None:
    """
    建立新的词汇字典和词向量矩阵
    :param type_vec_path: 二进制词向量文件路径
    :param type_word_path: 包含单词标签的文件路径
    :param final_vec_path: 保存最终词向量矩阵的文件路径
    :param final_word_path: 保存最终词汇表的路径
    """
    # 加载二进制文件
    model = KeyedVectors.load(type_vec_path, mmap='r')

    with open(type_word_path, 'r') as f:
        total_words = eval(f.read())

    # 词汇表：133961，词向量：133961
    word_dict = ['PAD', 'SOS', 'EOS', 'UNK']  # 0: PAD_ID, 1: SOS_ID, 2: EOS_ID, 3: UNK_ID
    fail_words = []
    rng = np.random.RandomState(None)
    pad_embedding = np.zeros(shape=(1, 300)).squeeze()
    unk_embedding = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
    sos_embedding = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
    eos_embedding = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
    word_vectors = [pad_embedding, sos_embedding, eos_embedding, unk_embedding]

    for word in total_words:
        try:
            word_vectors.append(model.wv[word])  # 加载词向量
            word_dict.append(word)
        except:
            print(f"无法找到单词 '{word}'")
            fail_words.append(word)

    # 输出有关词汇表中单词数量、词向量矩阵大小以及无法找到的单词数量的信息
    print(f"词汇表中单词总数：{len(word_dict)}")
    print(f"词向量矩阵大小：{len(word_vectors)}")
    print(f"未找到单词数：{len(fail_words)}")

    word_vectors = np.array(word_vectors)
    word_dict = dict(map(reversed, enumerate(word_dict)))

    with open(final_vec_path, 'wb') as file:
        pickle.dump(word_vectors, file)

    with open(final_word_path, 'wb') as file:
        pickle.dump(word_dict, file)

    print("已完成建立词汇字典和词向量矩阵。")


def get_index(text_type: str, text: List[str], word_dict: dict) -> List[int]:
    """
    在词汇字典中查找单词索引
    :param text_type: 文本类型('code' 或 'text')
    :param text: 输入的文本
    :param word_dict: 词汇字典
    :return: 单词索引列表
    """
    location = []

    if text_type == 'code':
        location.append(1)
        len_c = len(text)

        if len_c + 1 < 350:
            if len_c == 1 and text[0] == '-1000':
                location.append(2)
            else:
                for i in range(0, len_c):
                    if word_dict.get(text[i]) is not None:
                        index = word_dict.get(text[i])
                        location.append(index)
                    else:
                        index = word_dict.get('UNK')
                        location.append(index)

                location.append(2)
        else:
            for i in range(0, 348):
                if word_dict.get(text[i]) is not None:
                    index = word_dict.get(text[i])
                    location.append(index)
                else:
                    index = word_dict.get('UNK')
                    location.append(index)
            location.append(2)
    else:
        if len(text) == 0:
            location.append(0)
        elif text[0] == '-10000':
            location.append(0)
        else:
            for i in range(0, len(text)):
                if word_dict.get(text[i]) is not None:
                    index = word_dict.get(text[i])
                    location.append(index)
                else:
                    index = word_dict.get('UNK')
                    location.append(index)

    return location


def serialize(word_dict_path: str, type_path: str, final_type_path: str) -> None:
    """
    将训练数据、测试数据和验证数据序列化
    :param word_dict_path: 词汇字典文件路径
    :param type_path: 训练/测试/验证数据文件路径
    :param final_type_path: 保存最终序列化数据的路径
    """
    with open(word_dict_path, 'rb') as f:
        word_dict = pickle.load(f)

    with open(type_path, 'r') as f:
        corpus = eval(f.read())

    total_data = []

    for i in range(0, len(corpus)):
        qid = corpus[i][0]

        Si_word_list = get_index('text', corpus[i][1][0], word_dict)
        Si1_word_list = get_index('text', corpus[i][1][1], word_dict)

        tokenized_code = get_index('code', corpus[i][2][0], word_dict)

        query_word_list = get_index('text', corpus[i][3], word_dict)

        block_length = 4
        label = 0

        if len(Si_word_list) > 100:
            Si_word_list = Si_word_list[:100]
        else:
            for k in range(0, 100 - len(Si_word_list)):
                Si_word_list.append(0)

        if len(Si1_word_list) > 100:
            Si1_word_list = Si1_word_list[:100]
        else:
            for k in range(0, 100 - len(Si1_word_list)):
                Si1_word_list.append(0)

        if len(tokenized_code) < 350:
            for k in range(0, 350 - len(tokenized_code)):
                tokenized_code.append(0)
        else:
            tokenized_code = tokenized_code[:350]

        if len(query_word_list) > 25:
            query_word_list = query_word_list[:25]
        else:
            for k in range(0, 25 - len(query_word_list)):
                query_word_list.append(0)

        one_data = [qid, [Si_word_list, Si1_word_list], [tokenized_code], query_word_list, block_length, label]
        total_data.append(one_data)

    with open(final_type_path, 'wb') as file:
        pickle.dump(total_data, file)


def get_new_dict_append(type_vec_path: str, previous_dict: str, previous_vec: str, append_word_path: str,
                        final_vec_path: str, final_word_path: str) -> None:
    """加载原有词典和词向量，追加要添加的词，保存新的词典和词向量"""
    # 加载原先的词典和词向量
    with open(previous_dict, 'rb') as f:
        pre_word_dict = pickle.load(f)

    with open(previous_vec, 'rb') as f:
        pre_word_vec = pickle.load(f)

    # 加载要追加的词典文件
    with open(append_word_path, 'r', encoding='utf-8') as f:
        append_word = f.readlines()

    # 获取找到的词向量和无法找到的词
    word_vectors_list, fail_word_list = load_word_vectors(type_vec_path, pre_word_dict, pre_word_vec)
    model = KeyedVectors.load(type_vec_path, mmap='r')
    # 追加找到的词至列表中
    for word in map(str.strip, append_word):
        try:
            word_vectors_list.append(model.wv[word])
            pre_word_dict[word] = len(pre_word_dict)
        except KeyError:
            print(f'Cannot find embedding for word "{word}"')
        except ValueError:
            print(f'Cannot serialize object: {word}')

    # 保存新词典和词向量
    word_vectors = np.array(word_vectors_list)
    with open(final_vec_path, 'wb') as f:
        pickle.dump(word_vectors, f, protocol=4)

    with open(final_word_path, 'wb') as f:
        pickle.dump(pre_word_dict, f, protocol=4)

    print("Completed")


if __name__ == '__main__':
    # 配置路径
    path_dict = {
        'python_bin': '../hnn_process/embeddings/10_10/python_struc2vec.bin',
        'sql_bin': '../hnn_process/embeddings/10_8_embeddings/sql_struc2vec.bin',
        'python_word_dict': '../hnn_process/embeddings/python/python_word_dict_final.pkl',
        'python_word_vec': '../hnn_process/embeddings/python/python_word_vocab_final.pkl',
        'sql_word_dict': '../hnn_process/embeddings/sql/sql_word_dict_final.pkl',
        'sql_word_vec': '../hnn_process/embeddings/sql/sql_word_vocab_final.pkl',
        'final_word_dict_python': '../hnn_process/ulabel_data/python_word_dict.txt',
        'final_word_dict_sql': '../hnn_process/ulabel_data/sql_word_dict.txt',
        'new_python_large': '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlable.txt',
        'new_sql_large': '../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.txt',
        'python_final_word_vec': '../hnn_process/ulabel_data/large_corpus/python_word_vocab_final.pkl',
        'python_final_word_dict': '../hnn_process/ulabel_data/large_corpus/python_word_dict_final.pkl',
        'sql_final_word_vec': '../hnn_process/ulabel_data/large_corpus/sql_word_vocab_final.pkl',
        'sql_final_word_dict': '../hnn_process/ulabel_data/large_corpus/sql_word_dict_final.pkl'
    }

    # 追加新词到 Python 和 SQL 词典中
    get_new_dict_append(path_dict['python_bin'], path_dict['python_word_dict'], path_dict['python_word_vec'],
                        path_dict['final_word_dict_python'], path_dict['python_final_word_vec'],
                        path_dict['python_final_word_dict'])
    get_new_dict_append(path_dict['sql_bin'], path_dict['sql_word_dict'], path_dict['sql_word_vec'],
                        path_dict['final_word_dict_sql'], path_dict['sql_final_word_vec'],
                        path_dict['sql_final_word_dict'])

    # 将未打标签的语料序列化并保存至本地
    serialize(path_dict['python_final_word_dict'], path_dict['new_python_large'],
              '../hnn_process/ulabel_data/large_corpus/multiple/seri_python_large_multiple_unlable.pkl')
    serialize(path_dict['sql_final_word_dict'], path_dict['new_sql_large'],
              '../hnn_process/ulabel_data/large_corpus/multiple/seri_sql_large_multiple_unlable.pkl')

    print('序列化完毕')