import math as m
class Circle:
    def __init__(self,radius):
        self.__radius=radius
    @property
    def radius(self):
        return self.__radius
    @radius.setter
    def radius(self,radius):
        if(radius<=0):
            raise ValueError("cuole")
        self.__radius=radius
    
    @property
    def area(self):
        return m.pi*self.__radius*self.__radius
    @property
    def cir(self):
        return 2*m.pi*self.__radius
c=Circle(5)
print("半径{}".format(c.radius))
print("圆的面积{}".format(c.area))
print("圆的周长{}".format(c.cir))
c.radius=10
print("半径{}".format(c.radius))
print("圆的面积{}".format(c.area))
print("圆的周长{}".format(c.cir))
c.radius=-1