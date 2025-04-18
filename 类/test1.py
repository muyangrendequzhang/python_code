class User:
    def __init__(self,first_name,last_name,email,address):
        self.first_name=first_name
        self.last_name=last_name
        self.email=email
        self.address=address
        self.login_attempts=0
    def increment(self):
        self.login_attempts+=1
    def reset_login(self):
        self.login_attempts=0
        print("重置登入...")
    def print_login(self):
        print("登入次数{}".format(self.login_attempts))
    def describe_user(self):
        print("用户的真实姓名为"+self.first_name)
        print("用户名为"+self.last_name)
        print("email"+self.email)
        print("家庭地址"+self.address)
    def greet_user(self):
        print("欢迎回来，"+self.last_name)
class admin(User):
    class pri:
        def __init__(self,pri):
            self.pris=pri
        def show_pri(self):
            if(self.pris==1):
                print("管理员的权限：")
                print("重置密码")
                print("评论")
                print("管理其他用户")
    def __init__(self,first_name,last_name,email,address,pris):
        super().__init__(first_name,last_name,email,address)
        self.pris=pris
    def show_pri(self):
        if(self.pris==1):
            print("管理员的权限：")
            print("重置密码")
            print("评论")
            print("管理其他用户")
a2=admin.pri(1)
a1=admin("Eric","e_matt","e_examle@","xiamen",1)
a2.show_pri()
a1.describe_user()
a1.greet_user()
a1.show_pri()

