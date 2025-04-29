import jieba
from wordcloud import WordCloud
with open("第三方库/红楼梦.txt",encoding='utf-8') as f:
    txt=f.read()
words=jieba.lcut(txt)
connts={}
counts = {}
for word in words:
    if len(word) == 1:
        continue
    else:
        counts[word] = counts.get(word,0) + 1
excludes = ["什么","一个","我们","那里","你们","如今","只是","只得","丫头","说道","知道","老太太","起来","姑娘","这里","不敢","这些","出去","出来","他们","众人","自己","一面","太太","今儿","那些","贾珍","只见","怎么","奶奶","两个","没有","不是","不知","这个","听见","这样","进来","告诉","就是","东西","回来","咱们"]
for word in excludes:
    del(counts[word])
items = list(counts.items())
items.sort(key=lambda x:x[1], reverse=True)
newtxt=''.join(words)
WordCloud=WordCloud(font_path="msyh.ttc").generate(newtxt)
WordCloud.to_file("aa.png")