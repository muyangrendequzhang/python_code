john={"animal type":"python","name":"john","owner":"guido","weight":"43","eats":"bugs"}
clarence={"animal type":"chicken","name":"clarence","owner":"tiffany","weight":"2","eats":"seeds"}
peso={"animal type":"dog","name":"peso","owner":"eric","weight":"37","eats":"shoes"}
sum=[john,clarence,peso]
for dic in sum:
    for a,b in dic.items():
        print(a+":"+b)
    