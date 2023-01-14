class dog:

    def __init__(self,dogBreed="Bulldog",dogEyeColour="Green"):
        self.breed = dogBreed
        self.eyeColour = dogEyeColour

jake = dog()

print("This dog is a",jake.breed,"and it's eyes are",jake.eyeColour)