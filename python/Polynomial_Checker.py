class Polynomial(object):
    explanation = "I am a polynomial"

    def __init__(self, roots, leading_term):
        self.roots = roots
        self.leading_roots = leading_term
        self.order = len(roots)
    
    def display(self):
        string = str(self.leading_roots)
        for root in self.roots:
            if root == 0:
                string = string + "x"
            elif root > 0:
                string = string + "(x - {})".format(root)
            else:
                string = string + "(x + {})".format(-root)
        return string
    
    def multiply(self, other):
        roots = self.roots + other.roots
        leading_term = self.leading_roots * other.leading_term
        return Polynomial(roots, leading_term)
    
    def explain_to(self, caller):
        print("Hello, {}. {}.".format(caller, self.explanation))
        print("My roots are {}.".format(self.roots))


p = Polynomial((1,-2,6),-2)
q = Polynomial((3,6),4)
print(p)
r = p.multiply(q)
print(r.display())