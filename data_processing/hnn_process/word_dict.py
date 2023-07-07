import json
import pickle


def get_vocab(corpus1, corpus2):
    """
    构建初步词典

    :param corpus1: 第一个语料库
    :param corpus2: 第二个语料库
    :return: 词典
    """
    word_set = set()
    for corpus in [corpus1, corpus2]:
        for data in corpus:
            for sub_sub_data in [sub_data for sub_list in data[1:] for sub_data in sub_list]:
                word_set.update(sub_sub_data)
            word_set.update(data[3])

    print("词典大小: ", len(word_set))
    return word_set


def load_json(filename):
    """加载json文件"""
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


def load_pickle(filename):
    """加载pickle文件"""
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='iso-8859-1')
    return data


def save_json(data, filename):
    """保存为json文件"""
    with open(filename, "w") as f:
        json.dump(data, f)


def vocab_processing(filepath1, filepath2, save_path):
    """
    构建初步词典的主函数

    :param filepath1: 第一个语料库的路径
    :param filepath2: 第二个语料库的路径
    :param save_path: 保存词典的路径
    """
    total_data1 = load_json(filepath1)
    total_data2 = load_json(filepath2)

    word_set = get_vocab(total_data1, total_data2)

    save_json(list(word_set), save_path)


def final_vocab_processing(filepath1, filepath2, save_path):
    """
    构建最终词典的主函数

    :param filepath1: 初步词典的路径
    :param filepath2: 第二个语料库的路径
    :param save_path: 保存最终词典的路径
    """
    with open(filepath1, 'r') as f:
        total_data1 = set(load_json(f))
    with open(filepath2, 'r') as f:
        total_data2 = load_json(f)

    word_set = get_vocab(total_data1, total_data2) - total_data1

    print("初步词典大小: ", len(total_data1))
    print("最终词典大小: ", len(word_set))

    save_json(list(word_set), save_path)


if __name__ == "__main__":
    PYTHON_HNN = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/python_hnn_data_teacher.json'
    PYTHON_STAQC = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/staqc/python_staqc_data.json'
    PYTHON_WORD_DICT = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/word_dict/python_word_vocab_dict.json'

    SQL_HNN = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/sql_hnn_data_teacher.json'
    SQL_STAQC = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/staqc/sql_staqc_data.json'
    SQL_WORD_DICT = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/word_dict/sql_word_vocab_dict.json'










