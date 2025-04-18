import random
class Die:
    def __init__(self,sides):
        self.sides=sides
    def roll_die(self):
        print(random.randint(1,self.sides),end=" ")
    def print(self):
        print()
        print("10 rolls of a 6_sided die")
d1=Die(6)
for i in range(10):
    d1.roll_die()
d1.print()
d2=Die(10)
d3=Die(20)
for i in range(10):
    d2.roll_die()
d2.print()
for i in range(10):
    d3.roll_die()
d3.print()
