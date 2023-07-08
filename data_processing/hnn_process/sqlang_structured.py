# -*- coding: utf-8 -*-
import re
import sqlparse # 0.4.2
import inflection
# 词性还原
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
wordnet_ler = WordNetLemmatizer()

# 词干提取
from nltk.corpus import wordnet

#############################################################################
OTHER = 0
FUNCTION = 1
BLANK = 2
KEYWORD = 3
INTERNAL = 4

TABLE = 5
COLUMN = 6
INTEGER = 7
FLOAT = 8
HEX = 9
STRING = 10
WILDCARD = 11

SUBQUERY = 12

DUD = 13

ttypes = {0: "OTHER", 1: "FUNCTION", 2: "BLANK", 3: "KEYWORD", 4: "INTERNAL", 5: "TABLE", 6: "COLUMN", 7: "INTEGER",
          8: "FLOAT", 9: "HEX", 10: "STRING", 11: "WILDCARD", 12: "SUBQUERY", 13: "DUD", }

scanner = re.Scanner([(r"\[[^\]]*\]", lambda scanner, token: token), (r"\+", lambda scanner, token: "REGPLU"),
                      (r"\*", lambda scanner, token: "REGAST"), (r"%", lambda scanner, token: "REGCOL"),
                      (r"\^", lambda scanner, token: "REGSTA"), (r"\$", lambda scanner, token: "REGEND"),
                      (r"\?", lambda scanner, token: "REGQUE"),
                      (r"[\.~``;_a-zA-Z0-9\s=:\{\}\-\\]+", lambda scanner, token: "REFRE"),
                      (r'.', lambda scanner, token: None), ])


class SqlangParser:
    def __init__(self, sql, regex=False, rename=True):
        self.sql = self.sanitize_sql(sql)
        self.regex = regex

        self.parseTreeSentinel = False
        self.tableStack = []

        self.parse = sqlparse.parse(self.sql)

        self.idMap = {
            COLUMN: {},
            TABLE: {}
        }
        self.idMapInv = {}
        self.idCount = {
            COLUMN: 0,
            TABLE: 0
        }
        self.tokens = []

        self.parse = [self.parse[0]]

        self.remove_whitespaces(self.parse[0])
        self.identify_literals(self.parse[0])
        self.parse[0].ptype = SUBQUERY
        self.identify_subqueries(self.parse[0])
        self.identify_functions(self.parse[0])
        self.identify_tables(self.parse[0])

        self.parse_strings(self.parse[0])

        if rename:
            self.rename_identifiers(self.parse[0])

        self.tokens = self.get_tokens(self.parse)

    @staticmethod
    def sanitizeSql(sql):
        """
        This function sanitizes SQL so that delimiter
        doesn't create any problem while parsing
        """
        sql_uppered = sql.upper()
        if 'DELIMITER' in sql_uppered:
            found = False
            _start = idx = 0

            # If the query has delimiter statement,
            # we will remove new line characters
            for idx in range(len(sql)):
                if sql[idx] == '\'':
                    idx += 1
                    while sql[idx] != '\'':
                        idx += 1
                    continue
                if sql[idx] == '"' or sql[idx] == '`':
                    idx += 1
                    while sql[idx] != '"' and sql[idx] != '`':
                        idx += 1
                    continue
                if sql[idx:idx + 10] == 'DELIMITER ':
                    found = True
                    _start = idx + 10
                    break
            if found:
                delim = sql[_start:_start + 1]
                ret = ''
                for ch in sql:
                    if ch == delim:
                        continue
                    ret += ch
                return ret
        return sql

    def removeWhitespaces(self, tok):
        if isinstance(tok, sqlparse.sql.TokenList):
            tmp_children = []
            for c in tok.tokens:
                if not c.is_whitespace:
                    tmp_children.append(c)

            tok.tokens = tmp_children
            for c in tok.tokens:
                self.remove_whitespaces(c)

    def identifyLiterals(self, token_list):
        blank_tokens = [
            sqlparse.tokens.Name,
            sqlparse.tokens.Name.Placeholder
        ]
        blank_token_types = [sqlparse.sql.Identifier]

        for tok in token_list.tokens:
            if isinstance(tok, sqlparse.sql.TokenList):
                tok.ptype = INTERNAL
                self.identify_literals(tok)
            elif (tok.ttype == sqlparse.tokens.Keyword or
                  str(tok).lower() == "select"):
                tok.ttype = KEYWORD
            elif (tok.ttype in (sqlparse.tokens.Number.Integer,
                                sqlparse.tokens.Literal.Number.Integer)):
                tok.ttype = INTEGER
            elif (tok.ttype in (sqlparse.tokens.Number.Hexadecimal,
                                sqlparse.tokens.Literal.Number.Hexadecimal)):
                tok.ttype = HEX
            elif (tok.ttype in (sqlparse.tokens.Number.Float,
                                sqlparse.tokens.Literal.Number.Float)):
                tok.ttype = FLOAT
            elif (tok.ttype in [sqlparse.tokens.String.Symbol,
                                sqlparse.tokens.String.Single,
                                sqlparse.tokens.String.Double,
                                sqlparse.tokens.Literal.String.Single,
                                sqlparse.tokens.Literal.String.Symbol]):
                tok.ttype = STRING
            elif (tok.ttype == sqlparse.tokens.Wildcard):
                tok.ttype = WILDCARD
            elif (tok.ttype in blank_tokens or isinstance(tok, blank_token_types[0])):
                tok.ttype = COLUMN

    def identifyFunctions(self, token_list):
        for tok in token_list.tokens:
            if isinstance(tok, sqlparse.sql.Function):
                self.parseTreeSentinel = True
            elif isinstance(tok, sqlparse.sql.Parenthesis):
                self.parseTreeSentinel = False
            if self.parseTreeSentinel:
                tok.ttype = FUNCTION
            if isinstance(tok, sqlparse.sql.TokenList):
                self.identify_functions(tok)

    def identifyTables(self, token_list):
        for tok in token_list.tokens:
            if isinstance(tok, sqlparse.sql.IdentifierList):
                for ix, tok_ in enumerate(tok.tokens):
                    if isinstance(tok_, sqlparse.sql.Identifier) and (
                            tok_.ttype == sqlparse.tokens.Name or tok_.ttype == sqlparse.tokens.Name.Schema):
                        table_name = tok_.get_parent_name()
                        self.idMap["TABLE"][table_name] = f"tab{self.idCount['TABLE']}"
                        self.idMapInv[self.idMap["TABLE"][table_name]] = table_name

                        tok_ix = tok.tokens[ix]
                        tok_ix.value = self.idMap["TABLE"][table_name]
                        self.idCount["TABLE"] += 1
            elif isinstance(tok, sqlparse.sql.Identifier) and (
                    tok.ttype == sqlparse.tokens.Name or tok.ttype == sqlparse.tokens.Name.Schema):
                table_name = tok.get_parent_name()
                if table_name not in self.idMap["TABLE"]:
                    self.idMap["TABLE"][table_name] = f"tab{self.idCount['TABLE']}"
                    self.idMapInv[self.idMap["TABLE"][table_name]] = table_name

                    tok.value = self.idMap["TABLE"][table_name]
                    self.idCount["TABLE"] += 1
            if isinstance(tok, sqlparse.sql.TokenList):
                self.identify_tables(tok)

    def identifySubqueries(self, token_list):
        is_sub_query = False

        for tok in token_list.tokens:
            if isinstance(tok, sqlparse.sql.TokenList):
                sub_query = self.identifySubqueries(tok)
                if (sub_query and isinstance(tok, sqlparse.sql.Parenthesis)):
                    tok.ttype = SUBQUERY
            elif str(tok).lower() == "select":
                is_sub_query = True
        return is_sub_query

    def renameIdentifiers(self, tok):
        if isinstance(tok, sqlparse.sql.TokenList):
            for c in tok.tokens:
                self.renameIdentifiers(c)
        elif tok.ttype == COLUMN:
            if str(tok) not in self.idMap[COLUMN]:
                col_name = f"col{self.idCount[COLUMN]}"
                self.idMap[COLUMN][str(tok)] = col_name
                self.idMapInv[col_name] = str(tok)
                self.idCount[COLUMN] += 1
            tok.value = self.idMap[COLUMN][str(tok)]
        elif tok.ttype in [TABLE, sqlparse.tokens.Name.Schema,
                           sqlparse.tokens.Name]:
            if str(tok) not in self.idMap[TABLE]:
                tab_name = f"tab{self.idCount[TABLE]}"
                self.idMap[TABLE][str(tok)] = tab_name
                self.idMapInv[tab_name] = str(tok)
                self.idCount[TABLE] += 1
            tok.value = self.idMap[TABLE][str(tok)]

        elif tok.ttype == FLOAT:
            tok.value = "CODFLO"
        elif tok.ttype == INTEGER:
            tok.value = "CODINT"
        elif tok.ttype == HEX:
            tok.value = "CODHEX"

    def parseStrings(self, parse: sqlparse.sql.ParsedSQL):
        for statement_no, statement in enumerate(parse.statements):
            string_tokens = []
            for tok in statement.tokens:
                if tok.ttype == sqlparse.tokens.String.Symbol or tok.ttype == sqlparse.tokens.String.Single or tok.ttype == sqlparse.tokens.String.Double or tok.ttype == sqlparse.tokens.Literal.String.Single or tok.ttype == sqlparse.tokens.Literal.String.Symbol:
                    string_tokens.append(tok)
            for st in string_tokens:
                st_wrap = sqlparse.parse(f'\'{st.value}\'')[0].tokens[0]
                st_wrap.parent = st.parent
                statement.tokens[statement.tokens.index(
                    st)] = st_wrap

    @staticmethod
    def getTokens(parse):
        flatParse = []
        for expr in parse:
            for token in expr.flatten():
                if token.ttype == STRING:
                    flatParse.extend(str(token).split(' '))
                else:
                    flatParse.append(str(token))
        return flatParse

    def removeWhitespaces(self, tok):
        """
        Remove whitespace tokens from the given token list and its children.

        Args:
            tok (sqlparse.sql.TokenList): SQL token list.
        """
        if isinstance(tok, sqlparse.sql.TokenList):
            tmp_children = []
            for c in tok.tokens:
                if not c.is_whitespace:
                    tmp_children.append(c)

            tok.tokens = tmp_children
            for c in tok.tokens:
                self.removeWhitespaces(c)

    def identifySubQueries(self, token_list):
        """
        Identify subquery tokens in the SQL token list and modify their ttypes.

        Args:
            token_list (sqlparse.sql.TokenList): SQL token list.

        Returns:
            bool: True if a subquery is found, False otherwise.
        """
        is_sub_query = False

        for tok in token_list.tokens:
            if isinstance(tok, sqlparse.sql.TokenList):
                sub_query = self.identifySubQueries(tok)
                if sub_query and isinstance(tok, sqlparse.sql.Parenthesis):
                    tok.ttype = SUBQUERY
            elif str(tok) == "select":
                is_sub_query = True
        return is_sub_query

    def identifyLiterals(self, token_list):
        """
        Identify and modify literal tokens in the SQL token list.

        Args:
            token_list (sqlparse.sql.TokenList): SQL token list.
        """
        blank_tokens = [sqlparse.tokens.Name, sqlparse.tokens.Name.Placeholder]
        blank_token_types = [sqlparse.sql.Identifier]

        for tok in token_list.tokens:
            if isinstance(tok, sqlparse.sql.TokenList):
                token.ptype = INTERNAL
                self.identifyLiterals(tok)
            elif tok.ttype == sqlparse.tokens.Keyword or str(tok) == "select":
                tok.ttype = KEYWORD
            elif (
                    tok.ttype == sqlparse.tokens.Number.Integer
                    or tok.ttype == sqlparse.tokens.Literal.Number.Integer
            ):
                tok.ttype = INTEGER
            elif (
                    tok.ttype == sqlparse.tokens.Number.Hexadecimal
                    or tok.ttype == sqlparse.tokens.Literal.Number.Hexadecimal
            ):
                tok.ttype = HEX
            elif (
                    tok.ttype == sqlparse.tokens.Number.Float
                    or tok.ttype == sqlparse.tokens.Literal.Number.Float
            ):
                tok.ttype = FLOAT
            elif (
                    tok.ttype == sqlparse.tokens.String.Symbol
                    or tok.ttype == sqlparse.tokens.String.Single
                    or tok.ttype == sqlparse.tokens.Literal.String.Single
                    or tok.ttype == sqlparse.tokens.Literal.String.Symbol
            ):
                tok.ttype = STRING
            elif tok.ttype == sqlparse.tokens.Wildcard:
                tok.ttype = WILDCARD
            elif tok.ttype in blank_tokens or isinstance(tok, blank_token_types[0]):
                tok.ttype = COLUMN

    def identifyFunctions(self, token_list):
        """
        Identify and modify function tokens in the SQL token list.

        Args:
            token_list (sqlparse.sql.TokenList): SQL token list.
        """
        for tok in token_list.tokens:
            if isinstance(tok, sqlparse.sql.Function):
                self.parse_tree_sentinel = True
            elif isinstance(tok, sqlparse.sql.Parenthesis):
                self.parse_tree_sentinel = False
            if self.parse_tree_sentinel:
                tok.ttype = FUNCTION
            if isinstance(tok, sqlparse.sql.TokenList):
                self.identifyFunctions(tok)

    def identifyTables(self, token_list):
        """
        Identify and modify table tokens in the SQL token list.

        Args:
            token_list (sqlparse.sql.TokenList): SQL token list.
        """
        if token_list.ptype == SUBQUERY:
            self.table_stack.append(False)

        for i in range(len(token_list.tokens)):
            prev_tok = token_list.tokens[i - 1]
            tok = token_list.tokens[i]

            if (
                    str(tok) == "."
                    and tok.ttype == sqlparse.tokens.Punctuation
                    and prev_tok.ttype == COLUMN
            ):
                prev_tok.ttype = TABLE
            elif str(tok) == "from" and tok.ttype == sqlparse.tokens.Keyword:
                self.table_stack[-1] = True
            elif (
                    str(tok) == "where"
                    or str(tok) == "on"
                    or str(tok) == "group"
                    or str(tok) == "order"
                    or str(tok) == "union"
            ) and tok.ttype == sqlparse.tokens.Keyword:
                self.table_stack[-1] = False

            if isinstance(tok, sqlparse.sql.TokenList):
                self.identify_tables(tok)
            elif tok.ttype == COLUMN:
                if self.table_stack[-1]:
                    tok.ttype = TABLE

        if token_list.ptype == SUBQUERY:
            self.table_stack.pop()

    def __str__(self):
        """
        Represent SQL token list as a string.

        Returns:
            str: String representation of the SQL token list.
        """
        return " ".join([str(tok) for tok in self.tokens])

    def parseSql(self):
        """
        Parse the SQL token list into a list of strings.

        Returns:
            list: List of strings containing the SQL tokens.
        """
        return [str(tok) for tok in self.tokens]


#############################################################################

#############################################################################
#缩略词处理
def revertAbbrev(line):
    pat_is = re.compile("(it|he|she|that|this|there|here)(\"s)", re.I)
    # 's
    pat_s1 = re.compile("(?<=[a-zA-Z])\"s")
    # s
    pat_s2 = re.compile("(?<=s)\"s?")
    # not
    pat_not = re.compile("(?<=[a-zA-Z])n\"t")
    # would
    pat_would = re.compile("(?<=[a-zA-Z])\"d")
    # will
    pat_will = re.compile("(?<=[a-zA-Z])\"ll")
    # am
    pat_am = re.compile("(?<=[I|i])\"m")
    # are
    pat_are = re.compile("(?<=[a-zA-Z])\"re")
    # have
    pat_ve = re.compile("(?<=[a-zA-Z])\"ve")

    line = pat_is.sub(r"\1 is", line)
    line = pat_s1.sub("", line)
    line = pat_s2.sub("", line)
    line = pat_not.sub(" not", line)
    line = pat_would.sub(" would", line)
    line = pat_will.sub(" will", line)
    line = pat_am.sub(" am", line)
    line = pat_are.sub(" are", line)
    line = pat_ve.sub(" have", line)

    return line

#获取词性
def getWordpos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

#---------------------子函数1：句子的去冗--------------------
def processNlLine(line):
    # 句子预处理
    line = revertAbbrev(line)
    line = re.sub('\t+', '\t', line)
    line = re.sub('\n+', '\n', line)
    line = line.replace('\n', ' ')
    line = line.replace('\t', ' ')
    line = re.sub(' +', ' ', line)
    line = line.strip()
    # 骆驼命名转下划线
    line = inflection.underscore(line)

    # 去除括号里内容
    space = re.compile(r"\([^\(|^\)]+\)")  # 后缀匹配
    line = re.sub(space, '', line)
    # 去除末尾.和空格
    line=line.strip()
    return line


#---------------------子函数1：句子的分词--------------------
def processSentWord(line):
    # 找单词
    line = re.findall(r"[\w]+|[^\s\w]", line)
    line = ' '.join(line)

    # 替换小数
    decimal = re.compile(r"\d+(\.\d+)+")
    line = re.sub(decimal, 'TAGINT', line)
    # 替换字符串
    string = re.compile(r'\"[^\"]+\"')
    line = re.sub(string, 'TAGSTR', line)
    # 替换十六进制
    decimal = re.compile(r"0[xX][A-Fa-f0-9]+")
    line = re.sub(decimal, 'TAGINT', line)
    # 替换数字 56
    number = re.compile(r"\s?\d+\s?")
    line = re.sub(number, ' TAGINT ', line)
    # 替换字符 6c60b8e1
    other = re.compile(r"(?<![A-Z|a-z|_|])\d+[A-Za-z]+")  # 后缀匹配
    line = re.sub(other, 'TAGOER', line)
    cut_words= line.split(' ')
    # 全部小写化
    cut_words = [x.lower() for x in cut_words]
    #词性标注
    word_tags = pos_tag(cut_words)
    tags_dict = dict(word_tags)
    word_list=[]
    for word in cut_words:
        word_pos = getWordpos(tags_dict[word])
        if word_pos in ['a', 'v', 'n', 'r']:
            # 词性还原
            word = wordnet_ler.lemmatize(word, pos=word_pos)
        # 词干提取(效果最好）
        word = wordnet.morphy(word) if wordnet.morphy(word) else word
        word_list.append(word)
    return word_list


#############################################################################

def filter_all_invachar(line):
    # 去除非常用符号；防止解析有误
    line = re.sub('[^(0-9|a-z|A-Z|\-|_|\'|\"|\-|\(|\)|\n)]+', ' ', line)
    # 包括\r\t也清除了
    # 中横线
    line = re.sub('-+', '-', line)
    # 下划线
    line = re.sub('_+', '_', line)
    # 去除横杠
    line = line.replace('|', ' ').replace('¦', ' ')
    return line

def filter_invalid_characters(line):
    #去除非常用符号；防止解析有误
    line = re.sub(r'[^(0-9|a-z|A-Z|\-|#|/|_|,|\'|=|>|<|\"|\-|\\|\(|\)|\?|\.|\*|\+|\[|\]|\^|\{|\}|\n)]+', ' ', line)
    #包括\r\t也清除了
    # 中横线
    line = re.sub('-+', '-', line)
    # 下划线
    line = re.sub('_+', '_', line)
    # 去除横杠
    line = line.replace('|', ' ').replace('¦', ' ')
    return line

def processNlLine(line):
    line = re.sub('\n+', ' ', line)
    line = re.sub('\t+', ' ', line)
    line = re.sub(' +', ' ', line)
    return line.strip()

def processSentWord(line):
    return re.findall(r"[\w]+|[^\s\w]", line)

def sqlangCodeParse(line):
    line = filter_invalid_characters(line)
    line = re.sub('\.+', '.', line)
    line = processNlLine(line)
    line = re.sub(r"\d+(\.\d+)+", 'number', line)
    line = line.strip()
    line = re.findall(r"[\w]+|[^\s\w]", line)
    line = ' '.join(line)

    try:
        query = SqlangParser(line, regex=True)
        typed_code = query.parseSql()[:-1]
        # 骆驼命名转下划线
        typed_code = inflection.underscore(' '.join(typed_code)).split(' ')
        # 列表里包含 '' 和' '
        cut_tokens = [re.sub("\s+", " ", x.strip()) for x in typed_code]
        # 全部小写化
        token_list = [x.lower() for x in cut_tokens]
        # 列表里包含 '' 或 ' '
        token_list = [x.strip() for x in token_list if x.strip() != '']
        # 返回列表
        return token_list
    except:
        return '-1000'

def sqlangQueryParse(line):
    line = filter_invalid_characters(line)
    line = processNlLine(line)
    word_list = processSentWord(line)
    # 分完词后,再去掉 括号
    for i in range(0, len(word_list)):
        if re.findall('[\(\)]', word_list[i]):
            word_list[i] = ''
    # 列表里包含 '' 或 ' '
    word_list = [x.strip() for x in word_list if x.strip() != '']
    return word_list

def sqlang_context_parse(line):
    line = filter_invalid_characters(line)
    line = processNlLine(line)
    word_list = processSentWord(line)
    # 列表里包含 '' 或 ' '
    word_list = [x.strip() for x in word_list if x.strip() != '']
    return word_list

if __name__ == '__main__':
    print(sqlangCodeParse('""geometry": {"type": "Polygon" , 111.676,"coordinates": [[[6.69245274714546, 51.1326962505233], [6.69242714158622, 51.1326908883821], [6.69242919794447, 51.1326955158344], [6.69244041615532, 51.1326998744549], [6.69244125953742, 51.1327001609189], [6.69245274714546, 51.1326962505233]]]} How to 123 create a (SQL  Server function) to "join" multiple rows from a subquery into a single delimited field?'))
    print(sqlangQueryParse("change row_height and column_width in libreoffice calc use python tagint"))
    print(sqlangQueryParse('MySQL Administrator Backups: "Compatibility Mode", What Exactly is this doing?'))
    print(sqlangCodeParse('>UPDATE Table1 \n SET Table1.col1 = Table2.col1 \n Table1.col2 = Table2.col2 FROM \n Table2 WHERE \n Table1.id =  Table2.id'))
    print(sqlangCodeParse("SELECT\n@supplyFee:= 0\n@demandFee := 0\n@charedFee := 0\n"))
    print(sqlangCodeParse('@prev_sn := SerialNumber,\n@prev_toner := Remain_Toner_Black\n'))
    print(sqlangCodeParse(' ;WITH QtyCTE AS (\n  SELECT  [Category] = c.category_name\n          , [RootID] = c.category_id\n          , [ChildID] = c.category_id\n  FROM    Categories c\n  UNION ALL \n  SELECT  cte.Category\n          , cte.RootID\n          , c.category_id\n  FROM    QtyCTE cte\n          INNER JOIN Categories c ON c.father_id = cte.ChildID\n)\nSELECT  cte.RootID\n        , cte.Category\n        , COUNT(s.sales_id)\nFROM    QtyCTE cte\n        INNER JOIN Sales s ON s.category_id = cte.ChildID\nGROUP BY cte.RootID, cte.Category\nORDER BY cte.RootID\n'))
    print(sqlangCodeParse("DECLARE @Table TABLE (ID INT, Code NVARCHAR(50), RequiredID INT);\n\nINSERT INTO @Table (ID, Code, RequiredID)   VALUES\n    (1, 'Physics', NULL),\n    (2, 'Advanced Physics', 1),\n    (3, 'Nuke', 2),\n    (4, 'Health', NULL);    \n\nDECLARE @DefaultSeed TABLE (ID INT, Code NVARCHAR(50), RequiredID INT);\n\nWITH hierarchy \nAS (\n    --anchor\n    SELECT  t.ID , t.Code , t.RequiredID\n    FROM @Table AS t\n    WHERE t.RequiredID IS NULL\n\n    UNION ALL   \n\n    --recursive\n    SELECT  t.ID \n          , t.Code \n          , h.ID        \n    FROM hierarchy AS h\n        JOIN @Table AS t \n            ON t.RequiredID = h.ID\n    )\n\nINSERT INTO @DefaultSeed (ID, Code, RequiredID)\nSELECT  ID \n        , Code \n        , RequiredID\nFROM hierarchy\nOPTION (MAXRECURSION 10)\n\n\nDECLARE @NewSeed TABLE (ID INT IDENTITY(10, 1), Code NVARCHAR(50), RequiredID INT)\n\nDeclare @MapIds Table (aOldID int,aNewID int)\n\n;MERGE INTO @NewSeed AS TargetTable\nUsing @DefaultSeed as Source on 1=0\nWHEN NOT MATCHED then\n Insert (Code,RequiredID)\n Values\n (Source.Code,Source.RequiredID)\nOUTPUT Source.ID ,inserted.ID into @MapIds;\n\n\nUpdate @NewSeed Set RequiredID=aNewID\nfrom @MapIds\nWhere RequiredID=aOldID\n\n\n/*\n--@NewSeed should read like the following...\n[ID]  [Code]           [RequiredID]\n10....Physics..........NULL\n11....Health...........NULL\n12....AdvancedPhysics..10\n13....Nuke.............12\n*/\n\nSELECT *\nFROM @NewSeed\n"))
