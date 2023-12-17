from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from PIL import Image
import pickle
import io
import numpy as np

app = FastAPI()

# Load the trained model
with open("final_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define the endpoint for image classification
@app.post("/predict/")
async def predict_single_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    image = image.resize((64, 64))  # Resize image to match the model's input size
    img_array = np.array(image)
    img_array = img_array.reshape(1, 64, 64, 3)  # Reshape to match the model's input shape

    # Make predictions using the loaded model
    prediction = model.predict(img_array)

    # Conditions
    if prediction == 0:
        class_label = "Butterfly"
    elif prediction == 1:
        class_label = "Camel"
    elif prediction == 2:
         class_label = "Deer"
    elif prediction == 3:
         class_label = "Eagle"
    elif prediction == 4:
         class_label = "Fish"
    elif prediction == 5:
         class_label = "Giraffe"
    elif prediction == 6:
         class_label = "Hamster"
    elif prediction == 7:
         class_label = "Jaguar"
    elif prediction == 8:
         class_label = "Polar Bear"
    elif prediction == 9:
         class_label = "Zebra"

    return {"prediction": class_label}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)