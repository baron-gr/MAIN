
# class Dog:  ## Class
    
#     def __init__(self, name, age):       ## Initialization method
#         self.name = name        ## Attribute
#         self.age = age
    
#     def add_one(self, x):       ## Method
#         return x + 1
    
#     def bark(self):
#         print("Bark")
        
#     def get_name(self):
#         return self.name
    
#     def get_age(self):
#         return self.age
    
#     def set_age(self, age):
#         self.age = age

# d = Dog("Tim", 34)
# d.set_age(23)
# print(d.get_age())


class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade  # 0 - 100
    
    def get_grade(self):
        return self.grade

class Course:
    def __init__(self, name, max_students):
        self.name = name
        self.max_students = max_students
        self.students = []
    
    def add_student(self, student):
        if len(self.students) < self.max_students:
            self.students.append(student)
            return True
        return False
    
    def get_average_grade(self):
        value = 0
        for student in self.students:
            value += student.get_grade()
        return value / len(self.students)

s1 = Student("Tim", 19, 95)
s2 = Student("Bill", 19, 75)
s3 = Student("Jill", 19, 65)

course = Course("Science", 2)
course.add_student(s1)
course.add_student(s2)
# print(course.students[0].name)
# print(course.add_student(s3))
# print(course.get_average_grade())


## Inheritance
class Pet:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def show(self):
        print(f"I am {self.name} and I am {self.age} years old")
    
    def speak(self):
        print("Idk what I say")

class Cat(Pet):     ## Subclass inherits from super/parent class
    def __init__(self, name, age, color):       ## Adding new attribute not in super/parent class
        super().__init__(name, age)     ## Pull attributes from super/parent class
        self.color = color
    
    def speak(self):
        print("Meow")

    def show(self):
        print(f"I am {self.name} and I am {self.age} years old and I am {self.color}")

class Dog(Pet):
    def speak(self):
        print("Bark")

# p = Pet("Tim", 19)
# p.show()
# p.speak()
# c = Cat("Bill", 34, "Brown")
# c.show()
# c.speak()
# d = Dog("Jill", 24)
# d.speak()
# d.show()


## Class attributes
class Person:
    num_people = 0      ## Global attribute
    
    def __init__(self, name):
        self.name = name        ## Local attribute
        Person.add_person()
    
    @classmethod            ## Class methods
    def num_people_(cls):
        return cls.num_people
    
    @classmethod
    def add_person(cls):
        cls.num_people += 1

# p1 = Person("tim")
# print(Person.num_people)
# p2 = Person("jill")
# print(Person.num_people)
# print(p2.num_people)
# Person.num_people = 8
# print(Person.num_people)

# p1 = Person("tim")
# p2 = Person("jill")
# print(Person.num_people_())

## Static methods
class Math:
    
    @staticmethod       ## don't change anything
    def add5(x):
        return x + 5
    
    @staticmethod
    def pr():
        print("run")

print(Math.add5(5))
Math.pr()

