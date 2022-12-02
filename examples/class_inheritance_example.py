# 상속 (inheritance) 에서 *args, **kwargs example

class addition:
    # Superclass 
    def __init__(self, a, b):
        self.a = a
        self.b = b
    
    def add(self):
        return self.a + self.b


class calculator(addition):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cal = " I am a Calulator"


def main():
    
    tool_1 = calculator(1, 2)       # args
    print(tool_1.add())             # 3 ; (self.a = 1, self.b = 2)

    tool_2 = calculator(a=1, b=2)   # kwargs(keward arguments)
    print(tool_2.add())             # 3 ; (self.a = 1, self.b = 2)


if __name__ == "__main__":
    main()