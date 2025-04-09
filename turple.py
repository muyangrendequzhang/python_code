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