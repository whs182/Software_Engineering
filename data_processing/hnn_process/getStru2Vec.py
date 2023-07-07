
import pickle
from multiprocessing import Pool as ThreadPool

from python_structured import python_query_parse, python_code_parse, python_context_parse
from sqlang_structured import sqlang_query_parse, sqlang_code_parse, sqlang_context_parse


def parse_data(lang_type, data_list, split_num):
    """
    解析数据，返回解析后的数据列表
    :param lang_type: 语言类型，支持python或sql
    :param data_list: 数据列表, 格式为[qid, acontent, code_content, query_content]
    :param split_num: 每次进行解析的数据量
    :return: 解析后的数据列表，格式为[qid, [acontent1, acontent2], [code_content], query_content]
    """
    # 取出需要解析的数据
    acont1_data = [i[1][0][0] for i in data_list]
    acont2_data = [i[1][1][0] for i in data_list]
    query_data = [i[3][0] for i in data_list]
    code_data = [i[2][0][0] for i in data_list]
    qids = [i[0] for i in data_list]

    # 把数据分割成指定大小的列表，方便多进程处理
    acont1_split_list = [acont1_data[i:i + split_num] for i in range(0, len(acont1_data), split_num)]
    acont2_split_list = [acont2_data[i:i + split_num] for i in range(0, len(acont2_data), split_num)]
    query_split_list = [query_data[i:i + split_num] for i in range(0, len(query_data), split_num)]
    code_split_list = [code_data[i:i + split_num] for i in range(0, len(code_data), split_num)]

    # 根据不同的语言类型，选择相应的解析函数进行解析
    if lang_type == 'python':
        context_parse_fn = python_context_parse
        query_parse_fn = python_query_parse
        code_parse_fn = python_code_parse

    if lang_type == 'sql':
        context_parse_fn = sqlang_context_parse
        query_parse_fn = sqlang_query_parse
        code_parse_fn = sqlang_code_parse

    # 采用多进程对列表中的数据进行解析
    pool = ThreadPool(10)
    acont1_list = pool.map(context_parse_fn, acont1_split_list)
    acont2_list = pool.map(context_parse_fn, acont2_split_list)
    query_list = pool.map(query_parse_fn, query_split_list)
    code_list = pool.map(code_parse_fn, code_split_list)
    pool.close()
    pool.join()

    # 把解析后的数据合并到一起
    acont1_cut = []
    for p in acont1_list:
        acont1_cut += p
    print('acont1条数：%d' % len(acont1_cut))

    acont2_cut = []
    for p in acont2_list:
        acont2_cut += p
    print('acont2条数：%d' % len(acont2_cut))

    query_cut = []
    for p in query_list:
        query_cut += p
    print('query条数：%d' % len(query_cut))

    code_cut = []
    for p in code_list:
        code_cut += p
    print('code条数：%d' % len(code_cut))

    # 把处理后的数据组织成列表返回
    total_data = []
    for i in range(0, len(qids)):
        total_data.append([qids[i], [acont1_cut[i], acont2_cut[i]], [code_cut[i]], query_cut[i]])
    return total_data


def process_data(lang_type, split_num, source_path, save_path):
    """
    对数据进行处理，调用parse_data函数进行解析，最后保存解析后的数据
    :param lang_type: 语言类型，支持python或sql
    :param split_num: 每次进行解析的数据量
    :param source_path: 数据文件路径
    :param save_path: 保存文件路径
    :return: 无返回值
    """
    total_data = []
    with open(source_path, "rb") as f:
        corpus_list = pickle.load(f)

        total_data = parse_data(lang_type, corpus_list, split_num)

    with open(save_path, "w") as f:
        f.write(str(total_data))


if __name__ == '__main__':
    python_type = 'python'
    sqlang_type = 'sql'
    split_num = 1000

    staqc_python_path = '../hnn_process/ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_save = '../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.txt'

    staqc_sql_path = '../hnn_process/ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_save = '../hnn_process/ulabel_data/staqc/sql_staqc_unlabled_data.txt'

    process_data(sqlang_type, split_num, staqc_sql_path, staqc_sql_save)
    process_data(python_type, split_num, staqc_python_path, staqc_python_save)