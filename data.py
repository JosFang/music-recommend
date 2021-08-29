import random

def read_corpus(corpus_path, trian_proportion):
    with open(corpus_path, 'r', encoding='utf-8') as fr:
        data = fr.read().strip().split('\n')
    data = [d.split('\t') for d in data]
    # data = data[:10]
    data = [[int(d[0]), int(d[1]), int(d[2]), int(d[3]),
             [int(i) for i in d[4].strip().split(',')],
             [int(i) for i in d[5].strip().split(',')],
             [int(i) for i in d[6].strip().split(',')],
             [int(i) for i in d[7].strip().split(',')]] for d in data]
    user_id_musics_id_dic = {d[0]: d[6] for d in data}
    # user_id_musics_id_dic = {}
    # for d in data:
    #     t = d[6].strip().split(',')
    #     print(d)
    #     print(t)
    #     for i in t:
    #         if i == '':
    #             print(i)
    #             print("check")
    #     user_id_musics_id_dic[int(d[0])] = [int(i) for i in t]


    music_frequency = {}
    for m in user_id_musics_id_dic.values():
        for mm in m:
            if mm not in music_frequency:
                music_frequency[mm] = 0
            music_frequency[mm] += 1
    music_frequency = sorted(list(music_frequency.items()), key=lambda s: -s[1])
    music_id_dic = {}
    for m in music_frequency:
        music_id_dic[m[0]] = len(music_id_dic)

    for u in user_id_musics_id_dic.keys():
        user_id_musics_id_dic[u] = [music_id_dic[m] for m in user_id_musics_id_dic[u]]

    music_num = len(music_id_dic)
    user_id_session = []
    for d in data:
        t = []
        for i in range(1, len(d[4])):
            t.append([d[0], d[4][i-1], d[4][i]])
        # print(t[-1])
        # print(len(user_id_musics_id_dic[d[0]]))
        user_id_session.extend(t)
    # 洗牌
    random.shuffle(user_id_session)
    len_session = len(user_id_session)
    interval = int(len_session * trian_proportion)
    return user_id_session[:interval], user_id_session[interval:], user_id_musics_id_dic, music_num


def batch_yield(data, batch_size, user_id_musics_id_dic):
    sess = []
    precursor = []
    for user_id, start, end in data:
        sess.append(user_id_musics_id_dic[user_id][start: end])
        precursor.append(user_id_musics_id_dic[user_id][start-5:start])
        if len(sess) == batch_size:
            yield sess, precursor
            sess = []
            precursor = []
    if len(sess) != 0:
        yield sess, precursor

def pad_sequences(sequences, maxl=None, pad_mark=0):
    """
    :param sequences: one batch sequences
    :param pad_mark:
    :return:
    """
    if maxl:
        max_len = maxl
    else:
        max_len = max(map(lambda x: len(x), sequences))
    # print("max_len:", max_len)
    seq_list, seq_len_list = [], []
    for seq in sequences:
        # 如果没有前驱，把最热的歌曲作为前驱
        if seq == []:
            seq = [0]
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list