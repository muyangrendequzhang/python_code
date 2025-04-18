import random
a=["黑","梅","方","桃"]
s=set()
count =0
while len(s)<4:
    count+=1
    a1=random.choice(a)
    s1=random.randint(1,13)
    print("进行了{}次".format(count))
    if(a1 not in s):
        print("{}:{}".format(a1,s1))
    s.add(a1)
    
    
