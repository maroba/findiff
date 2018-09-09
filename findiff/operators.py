
class Operator(object):
    pass


class UnaryOperator(Operator):
    pass


class BinaryOperator(Operator):
    pass


class Plus(Operator):

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def apply(self, fd, u):
        u_left = self.left.apply(fd, u)
        u_right = self.right.apply(fd, u)
        return u_left + u_right


class Multiply(Operator):

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def apply(self, fd, u):
        return self.left * self.right.apply(fd, u)

