import json
from collections import Counter

DATA_FILE = '../data.txt'
SINGLE_SAVE_PATH = '../singles.txt'
MULTIPLE_SAVE_PATH = '../multiples.txt'


def load_data(filepath):
    """加载数据"""
    with open(filepath, 'r') as f:
        return json.load(f)


def count_single_occurences(data, value):
    """计算单个值在列表中的出现次数"""
    return data.count(value)


def separate_single_and_multiple(data):
    """将单个值和多个值分离"""
    qids = [d[0] for d in data]
    counts = Counter(qids)

    singles = []
    multiples = []
    for d in data:
        if counts[d[0]] == 1:
            singles.append(d)
        else:
            multiples.append(d)

    return singles, multiples


def save_data(data, filepath):
    """保存数据"""
    with open(filepath, 'w') as f:
        f.write(str(data))


def process_data(data_file, single_save_path, multiple_save_path):
    """处理数据并保存单个值和多个值"""
    data = load_data(data_file)
    singles, multiples = separate_single_and_multiple(data)
    save_data(singles, single_save_path)
    save_data(multiples, multiple_save_path)


# 主程序入口
if __name__ == "__main__":
    PYTHON_STAQC_PATH = 'hnn_process/ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    PYTHON_STAQC_SINGLE_SAVE = 'hnn_process/ulabel_data/staqc/single/python_staqc_single.txt'
    PYTHON_STAQC_MULTIPLE_SAVE = 'hnn_process/ulabel_data/staqc/multiple/python_staqc_multiple.txt'

    SQL_STAQC_PATH = 'hnn_process/ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    SQL_STAQC_SINGLE_SAVE = 'hnn_process/ulabel_data/staqc/single/sql_staqc_single.txt'
    SQL_STAQC_MULTIPLE_SAVE = 'hnn_process/ulabel_data/staqc/multiple/sql_staqc_multiple.txt'

    # 处理Python STAQC 数据
    process_data(PYTHON_STAQC_PATH, PYTHON_STAQC_SINGLE_SAVE, PYTHON_STAQC_MULTIPLE_SAVE)

    # 处理SQL STAQC 数据
    process_data(SQL_STAQC_PATH, SQL_STAQC_SINGLE_SAVE, SQL_STAQC_MULTIPLE_SAVE)