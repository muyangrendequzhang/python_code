import jieba
with open("第三方库/test.txt",encoding='utf-8') as f:
    txt=f.read()
words=jieba.lcut(txt)
counts={}
emo=['哭','苦','悲伤']
happy=['笑',"爽","开心"]
ne=0
po=0
for word in words:
    if(word in emo):
        ne+=1
    if(word in happy):
        po+=1
print("积极{}".format(po))
print("消极{}".format(ne))
if ne>po:
    print("积极")
else:
    print("消极")