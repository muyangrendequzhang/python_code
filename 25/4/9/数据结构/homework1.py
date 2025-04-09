ans=0
poker=[1,2,3,4,5,6,7,8,9,10,11,12,13,1,2,3,4,5,6,7,8,9,10,11,12,13,1,2,3,4,5,6,7,8,9,10,11,12,13,1,2,3,4,5,6,7,8,9,10,11,12,13]
for i in range(0,52):
    for j in range(i+1,52):
        for k in range(j+1,52):
            for m in range(k+1,52):
                if(poker[i]+poker[j]+poker[k]+poker[m]==24):
                    ans+=1
print(ans)