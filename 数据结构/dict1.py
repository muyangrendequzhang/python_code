dic={'nile':'egypt','huang':'china','ming':'fujian'}

# 使用.items()方法同时获取键和值
for river, country in dic.items():
    print("The {} runs through {}".format(river, country))

# 打印所有值
for country in dic.values():
    print(country)