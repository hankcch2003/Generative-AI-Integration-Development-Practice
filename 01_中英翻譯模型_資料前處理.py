pairs = [
    ("i live in taipei", "我住在台北"),
    ("i love night markets", "我喜歡夜市"),
    ("taipei is beautiful", "台北很美"),
    ("i go to night markets", "我去夜市"),
    ("night markets", "夜市"),    
    ("i live in taipei and love night markets", "我住在台北而且喜歡夜市"),
    ("Yilan is a beautiful county with many delicious foods", "宜蘭是一個很美的縣市且有很多美食")
]

# 使用列表推導式從 pairs 中擷取所有的中文句子 (位於每個元組的索引 1)
all_words = [p[1] for p in pairs]
print(all_words)

print('嘗試將 all_words 內的中文字擷取出來儲存為一個 list')

# 初始化一個集合 (set)，利用集合「元素不重複」的特性來過濾重複字元
chars = set()

# 使用雙層迴圈：第一層遍歷每個句子，第二層遍歷句子中的每個中文字
for words in all_words:
    for ch in words:
        chars.add(ch) # 將字元加入集合，若字元已存在則會自動忽略以達到去除重複的效果

print(chars)

# 將去除重複的字元集合轉換回列表 (list) 格式，方便後續進行排序或索引操作
output_words = list(chars)
print(output_words)