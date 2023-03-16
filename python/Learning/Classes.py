import numpy as np


class Polynomial(object):
    explanation = "I am a polynomial"

    def __init__(self, roots, leading_term):
        self.roots = roots
        self.leading_term = leading_term
        self.order = len(roots)

    def display(self):
        string = str(self.leading_term)
        for root in self.roots:
            if root == 0:
                string = string + "x"
            elif root > 0:
                string = string + "(x-{})".format(root)
            else:
                string = string + "(x+{})".format(-root)
        return string

    def multiply(self, other):
        roots = self.roots + other.roots
        leading_term = self.leading_term * other.leading_term
        return Polynomial(roots, leading_term)

    def explain_to(self, caller):
        print("Hello, {}. {}.".format(caller,self.explanation))
        print("My roots are {}.".format(self.roots))


p_roots_1 = (1, 2, -3,4)
p_roots_2 = (1,1,-2)
p_leading_term_1 = 2
p_leading_term_2 = -1
poly1 = Polynomial(p_roots_1, p_leading_term_1)
poly2 = Polynomial(p_roots_2, p_leading_term_2)

poly_dis1 = Polynomial.display(poly1)
poly_dis2 = poly2.display()
poly_multiply = poly1.multiply(poly2)
poly_multiply_dis = poly_multiply.display()
# Polynomial.explain_to(poly1, "Jake")
# poly2.explain_to("Baron")


class Calculator:
    def __init__(self, x_input, y_input):
        self.x = x_input
        self.y = y_input
    
    def test_add(self):
        return self.x + self.y
    
    def test_subtract(self):
        return self.x - self.y
    
    def test_divide(self):
        return self.x / self.y
    
    def test_multiply(self):
        return self.x * self.y


a = 5
b = 2
calc = Calculator(a, b)
answer = calc.test_add()
print(answer)