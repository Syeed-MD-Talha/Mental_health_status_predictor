from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from joblib import load

# Initialize FastAPI
app = FastAPI()

# Jinja2 template directory
templates = Jinja2Templates(directory="templates")

# Load the saved model once at startup
model = load("decision_tree_model.joblib")

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    """
    Display the input form for prediction.
    """
    return templates.TemplateResponse("index.html", {"request": request, "result": None})


@app.post("/", response_class=HTMLResponse)
def form_post(
    request: Request,
    Age: int = Form(...),
    Sleep_hours: float = Form(...),
    Stress_level: int = Form(...),
    Exercise_frequency: int = Form(...),
    Social_interaction_frequency: int = Form(...),
    Gender: str = Form(...)
):
    """
    Handle form submission, run the model prediction, and display the results.
    """

    # Convert Gender to one-hot style features
    # The original model expects: [Age, Sleep_hours, Strees_level, Exercise_frequency,
    #  Social_interaction_frequency, Gender_Male, Gender_Other]
    # Adjust the code if your columns differ.
    if Gender.lower() == "male":
        gender_male = 1
        gender_other = 0
    elif Gender.lower() == "other":
        gender_male = 0
        gender_other = 1
    else:  # assume 'female'
        gender_male = 0
        gender_other = 0

    # Construct the feature vector
    new_data = [
        Age,
        Sleep_hours,
        Stress_level,
        Exercise_frequency,
        Social_interaction_frequency,
        gender_male,
        gender_other
    ]

    mental_health_mapping = {
        2: "Poor",
        1: "Fair",
        0: "Good"
    }

    predicted_numeric = model.predict([new_data])[0]  # e.g., 1
    predicted_label = mental_health_mapping[predicted_numeric]  # e.g., "Fair"

    # Render the same page but display the result
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": predicted_label
    })
