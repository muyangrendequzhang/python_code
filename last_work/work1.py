class Acount:
    def __init__(self,id,ba,an):
        self.id=id
        self.ba=ba
        self._an=an
    def seto(self):
        self.id=0
        self.ba=100
        self.an=0
    def getMR(self):
        return self._an/12
    def getM(self):
        return self._an*self.ba/12
    def withdraw(self,value):
        self.ba-=value
    def deposit(self,value):
        self.ba+=value
a=Acount(1122,20000,0.045)
a.withdraw(2500)
a.deposit(3000)
print("id:{}".format(a.id))
print("金额：{}".format(a.ba))
print("月利率：{}".format(a.getMR()))
print("月利息：{}".format(a.getM()))