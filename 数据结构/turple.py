a=("电影a",2020,"动作",8.5)
b=("电影b",2018,"科幻",7.9)
c=("电影c",2021,"冒险",9.0)
d=("电影d",2019,"动作",7.5)
e=("电影e",2025,"科幻",8.0)
films=(a,b,c,d,e)
for film in films:
    print(film)
print("以下电影评分高于8.0")
for film in films:
    if(film[3]>=8.0):
        print(film)
print("以下电影发布于2025年后")
for film in films:
    if(film[1]>=2025):
        print(film)
a1=list(a)
b1=list(b)
c1=list(c)
d1=list(d)
e1=list(e)
print("以下电影2021年上映")
film1=[a1,b1,c1,d1,e1]
for i in film1:
    if(i[1]==2021):
        print(i)
for i in range(0,5):
    for j in range(0,4):
        if(film1[j][3]<film1[j+1][3]):
            film1[j],film1[j+1]=film1[j+1],film1[j]
for film in film1:
    print(film)