# -*- coding：utf-8 -*-
import re
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from itertools import chain
from sklearn.externals import joblib

global model
model = joblib.load("CRFmodel_V0.2.model")

def q_to_b(q_str):
    """
    功能：非中文文字的全角转半角
    输入：一个字符串
    输出：半角字符串
    
    """
    b_str = ""
    for uchar in q_str:
        inside_code = ord(uchar)
        if inside_code == 12288:  
            inside_code = 32
        elif 65374 >= inside_code >= 65281:  
            inside_code -= 65248
        b_str += chr(inside_code)
    return b_str

def process_t(words):
    """
    功能：处理时间词，带有连续的时间词性的多个词进行合并
    输入：由带有词性标签的字符组成的列表
    输出：合并相邻并且带有'/t'词性标签的字符后的列表
    """
    pro_words = []
    index = 0
    temp = u''
    while True:
        word = words[index] if index < len(words) else u''
        if u'/t' in word:
            temp = temp.replace(u'/t', u'') + word
        elif temp:
            pro_words.append(temp)
            pro_words.append(word)
            temp = u''
        elif word:
            pro_words.append(word)
        else:
            break
        index += 1
    return pro_words

def process_nr(words):
    """
    功能：处理姓名词，带有连续的姓名词性的多个词进行合并
    输入：由带有词性标签的字符组成的列表
    输出：合并相邻并且带有'/nr'词性标签的字符后的列表
    """
    pro_words = []
    index = 0
    while True:
        word = words[index] if index < len(words) else u''
        if u'/nr' in word:
            next_index = index + 1
            if next_index < len(words) and u'/nr' in words[next_index]:
                pro_words.append(word.replace(u'/nr', u'') + words[next_index])
                index = next_index
            else:
                pro_words.append(word)
        elif word:
            pro_words.append(word)
        else:
            break
        index += 1
    return pro_words 

def process_k(words):
    """
    功能：处理大粒度分词，包含在'[]'内的词可以作为组合词，将其合并，去掉两侧的'[]'
    输入：由带有词性标签的字符组成的列表
    输出：合并后的列表
    """
    pro_words = []
    index = 0
    temp = u''
    while True:
        word = words[index] if index < len(words) else u''
        if u'[' in word:
            temp += re.sub(pattern=u'/[a-zA-Z]*', repl=u'', string=word.replace(u'[', u''))
        elif u']' in word:
            w = word.split(u']')
            temp += re.sub(pattern=u'/[a-zA-Z]*', repl=u'', string=w[0])
            pro_words.append(temp+u'/'+w[1])
            temp = u''
        elif temp:
            temp += re.sub(pattern=u'/[a-zA-Z]*', repl=u'', string=word)
        elif word:
            pro_words.append(word)
        else:
            break
        index += 1
    return pro_words
	
#_maps提供一个由词性到实体标签的映射
_maps = {u't': u'T',
         u'nr': u'PER',
         u'ns': u'LOC',
         u'nt': u'ORG'}

def pos_to_tag(p):
    """
    功能：词性提取实体标签读法
    输入：语料库中的词性字符，如't','nr'等
    输出：对应的实体标签
    """
    t = _maps.get(p, None)
    return t if t else u'O'

def tag_perform(tag, index):
    """
    功能：根据实体中不同单字的位置为标签加上'B'或'I',使用BIO模式
    输入：tag:实体标签
          index:该标签在对应词语的标签组中的位置
    输出：完整的分词标签
    """
    if index == 0 and tag != u'O':
        return u'B_{}'.format(tag)
    elif tag != u'O':
        return u'I_{}'.format(tag)
    else:
        return tag
    
def pos_perform(pos):
    """
    功能：对于同属名词的词性（'nr','ns','nt'），去除词性携带的标签先验知识
    输入：词性（'nr','ns','nt'等）
    输出：除去'n'的名词词性
    """
    if pos in _maps.keys() and pos != u't':
        return u'n'
    else:
        return pos
    
def init_sequence(words_list):
    """
    功能：初始化字序列、词性序列、实体标签序列 
    输入：嵌套的列表，例如：[['迈向/v'], ['充满/v'], ['希望/n'], ['的/u'], ['新/a']]
    输出：word_seq:去掉词性标签的词语列表的嵌套，例如：[['迈', '向'], 
                                                        ['充', '满'], 
                                                        ['希', '望'], 
                                                        ['的'], 
                                                        ['新']]
          tag_seq:对应word_tag中每个元素的实体标识列表的嵌套，例如：[['O', 'O'], 
                                                                     ['O', 'O'], 
                                                                     ['O', 'O'], 
                                                                     ['O'], 
                                                                     ['O']]
    """
    words_seq = [[word.split(u'/')[0] for word in words] for words in words_list]
    pos_seq = [[word.split(u'/')[1] for word in words] for words in words_list]
    tag_seq = [[pos_to_tag(p) for p in pos] for pos in pos_seq]
    pos_seq = [[[pos_seq[index][i] for _ in range(len(words_seq[index][i]))]
                for i in range(len(pos_seq[index]))] for index in range(len(pos_seq))]
    tag_seq = [[[tag_perform(tag_seq[index][i], w) for w in range(len(words_seq[index][i]))]
                for i in range(len(tag_seq[index]))] for index in range(len(tag_seq))]
    pos_seq = [[u'un']+[pos_perform(p) for pos in pos_seq for p in pos]+[u'un'] for pos_seq in pos_seq]
    tag_seq = [[t for tag in tag_seq for t in tag] for tag_seq in tag_seq]
    word_seq = [[w for word in word_seq for w in word] for word_seq in words_seq]
    return word_seq, tag_seq

def segment_by_window(words_list=None, window=3):
    """
    功能：对words_list中的每个元素，按window长度从第一个开始以步长为1进行窗口切分
    输入：words_list:列表，例如：['<BOS>', '迈', '向', '充', '满']
          window:切分长度，默认为3
    输出：二层嵌套的列表，例如：[['<BOS>', '迈', '向'], 
                                 ['迈', '向', '充'], 
                                 ['向', '充', '满']]
    """
    words = []
    begin, end = 0, window
    for _ in range(1, len(words_list)):
        if end > len(words_list): break
        words.append(words_list[begin:end])
        begin += 1
        end += 1
    return words

def extract_feature(word_grams):
    """
    功能：特征模板，即对每个单字抽取其此模板定义的特征
    输入：三层列表的嵌套形式，并且加入词语的分隔标识符，例如：[[['<BOS>', '迈', '向'],
                                                                ['迈', '向', '充'],
                                                                ['向', '充', '满'],
                                                                ['充', '满', '希'],
                                                                ['满', '希', '望']]]
    输出：list of list of dicts，例如：[[{'bias': 1.0, 'w': '迈', 'w+1': '向', 'w-1': '<BOS>',
                                          'w-1:w': '<BOS>迈', 'w:w+1': '迈向'},
                                         {'bias': 1.0, 'w': '向', 'w+1': '充', 'w-1': '迈',
                                          'w-1:w': '迈向', 'w:w+1': '向充'},
                                         {'bias': 1.0, 'w': '充', 'w+1': '满', 'w-1': '向',
                                          'w-1:w': '向充', 'w:w+1': '充满'}, 
                                         {'bias': 1.0, 'w': '满', 'w+1': '希', 'w-1': '充',
                                          'w-1:w': '充满', 'w:w+1': '满希'},
                                         {'bias': 1.0, 'w': '希', 'w+1': '望', 'w-1': '满',
                                          'w-1:w': '满希', 'w:w+1': '希望'}]]
    """
    features, feature_list = [], []
    for index in range(len(word_grams)):
        for i in range(len(word_grams[index])):
            word_gram = word_grams[index][i]
            feature = {u'w-1': word_gram[0], u'w': word_gram[1], u'w+1': word_gram[2],
                       u'w-1:w': word_gram[0]+word_gram[1], u'w:w+1': word_gram[1]+word_gram[2],
                        # u'p-1': cls.pos_seq[index][i], u'p': cls.pos_seq[index][i+1],
                        # u'p+1': cls.pos_seq[index][i+2],
                        # u'p-1:p': cls.pos_seq[index][i]+cls.pos_seq[index][i+1],
                        # u'p:p+1': cls.pos_seq[index][i+1]+cls.pos_seq[index][i+2],
                        u'bias': 1.0}
            feature_list.append(feature)
        features.append(feature_list)
        feature_list = []
    return features
	
def predict(s):
    """
    功能：预测实体标签的函数
    输入：一段待检验的话
    输出：具有非'O'标签的词语及其实体类别
    """
    # 数据预处理部分 ## code starts here ##
    s = q_to_b(s)
    s = list(chain(*s))
    word_lists = [u'<BOS>']+s+[u'<EOS>']
    word_grams = [segment_by_window(word_lists)]
    X = extract_feature(word_grams)
    # 数据预处理部分 ## code ends here ##
    #
    # 预测 ## code starts here ##
    predict_list = model.predict(X)[0]
    # 预测 ## code ends here ##
    #
    # predict_list 中含有输入的段落中每一个字符（包括标点）的预测实体标签。留下有用实体的内容及标签，去掉'O'及其内容。
    #
    # 抽取有用内容及标签 ## code starts here
    entity_word_list = []
    entity_tag_list = []
    for i in range(len(predict_list)):
        if predict_list[i] != 'O':
            entity_word_list.append(s[i])
            entity_tag_list.append(predict_list[i])
    # 抽取有用内容及标签 ## code ends here
    #
    entity_word_list.append("END") #加上终止标识，为合并实体做准备
    entity_tag_list.append("END")  #加上终止标识，为合并实体标签做准备
    #
    # 实体及对应标签合并 ## code starts here ##
    entity_name = []
    entity_tags = []
    st = 0
    for j in range((len(entity_tag_list)-1)):
        if entity_tag_list[j] != entity_tag_list[j+1]:
            if r'B' not in entity_tag_list[j]:
                entity_name.append("".join(entity_word_list[st:j+1]))
                entity_tags.append(",".join(entity_tag_list[st:j+1]))
                st = j+1
    # 实体及对应标签合并 ## code ends here ##
    return entity_name, entity_tags
	
def print_result(entity_name, entity_tags):
    for i in range(len(entity_name)):
        length = 10 - len(entity_name[i].encode('gbk')) + len(entity_name[i])
        s = "实体内容：%-"+str(length)+"s\t,实体标签：%s"
        print(s % (entity_name[i],entity_tags[i]))
