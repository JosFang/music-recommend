# import time
#
# session_interval = 60*30
#
# with open("data/data_test.tsv", 'r', encoding='utf-8') as fr:
#     data = fr.read().strip().split('\n')
# print("read finish")
# data = [d.split('\t') for d in data]
#
# def time_transfor(daytime):
#     # daytime = "2009-05-04T13:06:10Z"
#     # 将日期转换成时间数组使用time.straptime(string, format)函数
#     timeArray = time.strptime(daytime, "%Y-%m-%dT%H:%M:%SZ")
#     # 转换成时间戳
#     timestamp = time.mktime(timeArray)
#     return int(timestamp)
#
# # user_id, music_artist = [], []
# #
# # for d in data:
# #     user_id.append(d[0])
# #     music_artist.append((d[5], d[3]))
# # user_id = set(user_id)
# # user_to_num = {u: int(u[5:])-1 for u in user_id}
# #
# # # 计算歌曲频数
# # music_num = {}
# # for m in music_artist:
# #     if m not in music_num:
# #         music_num[m] = 0
# #     music_num[m] += 1
# # print("歌曲总数：", len(music_num))
# # music_num = sorted(list(music_num.items()), key=lambda s: -s[1])
# # music_artist_to_num = {}
# # for m in music_num:
# #     music_artist_to_num[m[0]] = len(music_artist_to_num)
# #
# # final_data = []
# # for d in data:
# #     final_data.append((user_to_num[d[0]], time_transfor(d[1]), music_artist_to_num[(d[5], d[3])]))
# #
# # print(final_data[:4])
#
#
#
# # data = []
# # print("去重前：")
# # print("user_id数量：", len(user_id))
# # print("music_id数量：", len(music_id))
# # print("music数量：", len(music))
# # print("artist_name数量", len(artist_name))
# # print("artist_id数量", len(artist_id))
# #
# # print("整合中...")
# # user_id = set(user_id)
# # music_id = set(music_id)
# # music = set(music)
# # artist_name = set(artist_name)
# # artist_id = set(artist_id)
# # print("去重后")
# # print("user_id数量：", len(user_id))
# # print("music_id数量：", len(music_id))
# # print("music数量：", len(music))
# # print("artist_name数量", len(artist_name))
# # print("artist_id数量", len(artist_id))
#
# user_record = {}
# for d in data:
#     if d[0] not in user_record:
#         user_record[d[0]] = []
#     user_record[d[0]].append([time_transfor(d[1]), d[1], d[5]])
#
# # 按收听时间排序
# for u in user_record.keys():
#     user_record[u].sort(key=lambda s: s[0])
#     t_all = []
#     t = [user_record[u][0]]
#     for i in range(1, len(user_record[u])):
#         if user_record[u][i][0] - user_record[u][i-1][0] <= session_interval:
#             t.append(user_record[u][i])
#         else:
#             t_all.append(t)
#             t = [user_record[u][i]]
#     t_all.append(t)
#     user_record[u] = t_all
#
# artist_name_all = []
# music_name_all = []
# music_artist = []
# for d in data:
#     artist_name_all.append(d[3])
#     music_name_all.append(d[5])
#     if d[3] == "" or d[5] == "":
#         print("erorr!")
#     music_artist.append((d[5], d[3]))
#
# cont = 0
# for i in artist_name_all:
#     if i == "":
#         cont += 1
# print(cont)

# k = 10
# p = 11.8
# r = p / k
# F1 = (2*p*r)/(p+r)
# print(r)
# print(F1)

# import time
#
# session_interval = 60*30
#
# with open("data/data_test.tsv", 'r', encoding='utf-8') as fr:
#     data = fr.read().strip().split('\n')
# print("read finish")
# data = [d.split('\t') for d in data]
#
# def time_transfor(daytime):
#     # daytime = "2009-05-04T13:06:10Z"
#     # 将日期转换成时间数组使用time.straptime(string, format)函数
#     timeArray = time.strptime(daytime, "%Y-%m-%dT%H:%M:%SZ")
#     # 转换成时间戳
#     timestamp = time.mktime(timeArray)
#     return int(timestamp)
#
# # user_id, music_artist = [], []
# #
# # for d in data:
# #     user_id.append(d[0])
# #     music_artist.append((d[5], d[3]))
# # user_id = set(user_id)
# # user_to_num = {u: int(u[5:])-1 for u in user_id}
# #
# # # 计算歌曲频数
# # music_num = {}
# # for m in music_artist:
# #     if m not in music_num:
# #         music_num[m] = 0
# #     music_num[m] += 1
# # print("歌曲总数：", len(music_num))
# # music_num = sorted(list(music_num.items()), key=lambda s: -s[1])
# # music_artist_to_num = {}
# # for m in music_num:
# #     music_artist_to_num[m[0]] = len(music_artist_to_num)
# #
# # final_data = []
# # for d in data:
# #     final_data.append((user_to_num[d[0]], time_transfor(d[1]), music_artist_to_num[(d[5], d[3])]))
# #
# # print(final_data[:4])
#
#
#
# # data = []
# # print("去重前：")
# # print("user_id数量：", len(user_id))
# # print("music_id数量：", len(music_id))
# # print("music数量：", len(music))
# # print("artist_name数量", len(artist_name))
# # print("artist_id数量", len(artist_id))
# #
# # print("整合中...")
# # user_id = set(user_id)
# # music_id = set(music_id)
# # music = set(music)
# # artist_name = set(artist_name)
# # artist_id = set(artist_id)
# # print("去重后")
# # print("user_id数量：", len(user_id))
# # print("music_id数量：", len(music_id))
# # print("music数量：", len(music))
# # print("artist_name数量", len(artist_name))
# # print("artist_id数量", len(artist_id))
#
# user_record = {}
# for d in data:
#     if d[0] not in user_record:
#         user_record[d[0]] = []
#     user_record[d[0]].append([time_transfor(d[1]), d[1], d[5]])
#
# # 按收听时间排序
# for u in user_record.keys():
#     user_record[u].sort(key=lambda s: s[0])
#     t_all = []
#     t = [user_record[u][0]]
#     for i in range(1, len(user_record[u])):
#         if user_record[u][i][0] - user_record[u][i-1][0] <= session_interval:
#             t.append(user_record[u][i])
#         else:
#             t_all.append(t)
#             t = [user_record[u][i]]
#     t_all.append(t)
#     user_record[u] = t_all
#
# artist_name_all = []
# music_name_all = []
# music_artist = []
# for d in data:
#     artist_name_all.append(d[3])
#     music_name_all.append(d[5])
#     if d[3] == "" or d[5] == "":
#         print("erorr!")
#     music_artist.append((d[5], d[3]))
#
# cont = 0
# for i in artist_name_all:
#     if i == "":
#         cont += 1
# print(cont)

# k = 10
# p = 11.8
# r = p / k
# F1 = (2*p*r)/(p+r)
# print(r)
# print(F1)

# nums = input().strip().split(',')
# def is_shushu(n):
#     if n <= 3:
#         return True
#     else:
#         gen = int(n**0.5)
#         for i in range(2, gen+1):
#             if n % i == 0:
#                 return False
#         return True
# nums = [int(i) for i in nums]
#
# nums = sorted([i for i in nums if is_shushu(i)])
# if len(nums) == 0:
#     print("empty")
# else:
#     nums = [str(i) for i in nums]
#     print(','.join(nums))

# n = int(input())//10
# v = [0, 30, 50, 62, 37, 40, 45, 38, 15]
# w = [0, 2, 3, 5, 3, 5, 3, 4, 1]
# nf = [0] * (n+1)
# for i in range(1, len(v)):
#     for j in range(1, n+1)[::-1]:
#         if j >= w[i]:
#             nf[j] = max(nf[j], v[i] + nf[j-w[i]])
#         print(i, j, nf[j])
# print(nf[-1]*10)

import time
timeStamp = 1572936424076
timeStamp = int(timeStamp/1000)
timeArray = time.localtime(timeStamp)
otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
print(otherStyleTime)   # 2013--10--10 23:40:00









