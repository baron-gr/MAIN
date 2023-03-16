from dotenv import load_dotenv
load_dotenv()
import replicate

model = replicate.models.get("tencentarc/gfpgan")
version = model.versions.get("9283608cc6b7be6b65a8e44983db012355fde4132009bf99d976b2f0896856a3")

def predict_image(filename):
    inputs = {
        'img': open(filename, 'rb'),
        'version': "v1.4",
        'scale': 2,
    }
    print(inputs)

    output = version.predict(**inputs)
    print(output)
    return output