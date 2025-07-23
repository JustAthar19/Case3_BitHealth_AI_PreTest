import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI # type: ignore
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
from typing import List
import os
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Hospital Triage System",
    description="A FastAPI service to recommend medical departments based on patient symptoms using an LLM.",
    version="1.0.0"
)

# Hide myAPI key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Pydantic Model for input JSON
class PatientInput(BaseModel):
    gender: str
    age: int
    symptoms: List[str]

# Pydantic Model for Output JSON
class RecommendationOutput(BaseModel):
    recommended_department: str

class StatsOutput(BaseModel):
    department_counts: dict

# Rule Based Fallback -> Back Up mechanism incase the LLMs Failed
SYMPTOM_TO_DEPT = {
    "pusing": "Neurology",
    "sakit kepala": "Neurology",
    "sulit berjalan": "Neurology",
    "kehilangan keseimbangan": "Neurology",
    "mual": "Gastroenterology",
    "sakit perut": "Gastroenterology",
    "batuk": "Pulmonology",
    "sesak napas": "Pulmonology",
    "susah tidur": "Psychiatry",
    "menggigil": "Internal Medicine",
    "memar di tangan": "Dermatology",
    "gusi berdarah": "Dentistry"
}

try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-001",
        google_api_key=GOOGLE_API_KEY
    )
except Exception as e:
    print(f"Failed to initialize LLM: {e}")
    llm = None


prompt = PromptTemplate(
    input_variables=["gender", "age", "symptoms"],
    template="""
    You are a medical triage assistant. Given a patient with:
    - Gender: {gender}
    - Age: {age}
    - Symptoms: {symptoms}
    Recommend the most appropriate medical department (e.g., Neurology, Cardiology, Gastroenterology, etc.).
    Return only the department name, nothing else.
    If unsure, return 'General Medicine'.
    """   
)

if llm:
    chain = RunnableSequence(prompt | llm)
else:
    chain = None

@app.post("/recommend", response_model=RecommendationOutput)
async def recommned_department(patient: PatientInput):
    if not patient.symptoms:
        raise HTTPException(status_code=400, detail="Symptoms list cannot be empty")
    
    symptoms_str = ", ".join(patient.symptoms)
    if chain:
        try:
            response = chain.invoke({
                "gender": patient.gender,
                "age": patient.age,
                "symptoms": symptoms_str
            })
            department = response.content.strip()
            valid_departments = [
                "Neurology", "Cardiology", "Gastroenterology", "Pulmonology",
                "Psychiatry", "Internal Medicine", "Dentistry", "Dermatology",
                "General Medicine"
            ]
            if department in valid_departments:
                return RecommendationOutput(recommended_department=department)
        except Exception as e:
            print(f"Error: {e}")
    
    # Rule-based fallback implementation
    departments = [SYMPTOM_TO_DEPT.get(symptom.lower(), "General Medicine") for symptom in patient.symptoms]
    # Choose the most common department or set it to default (General Medicine)
    most_common = max(set(departments), key=departments.count, default="General Mdicine")
    return RecommendationOutput(recommended_department=most_common)


# Case 3: End-to-End Mini Project
# Task: Extend your FastAPI service from Case 3 by adding a /stats endpoint to summarize patient data, testing your ability to handle API logic and integrate with the existing system.

# Input: Reuse main.py from Case 3 (/Users/atharfathana/Documents/Machine Learning-AI/BitHeatlth-AI-PreTest/Case3_BitHealth_AI_PreTest/main.py).
# Requirement: Add a /stats POST endpoint that accepts a list of patient data (gender, age, symptoms) and returns the count of patients per recommended department.
# Challenge: Use the existing SYMPTOM_TO_DEPARTMENT mapping (rule-based) for recommendations and handle invalid inputs.

# @app.post("/stats", response_model=StatsOutput)
# async def department_starts(patients: List[PatientInput]):
#     logger.debug(f"Received /stats request with {len(patients)} patients: {patients}")
#     if not patients:
#         raise HTTPException(status_code=400, detail='Patient List Cannot Be Empty')
#     departments = []
#     for patient in patients:
#         # Skip the patient with empty symptoms
#         if not patient.symptoms:
#             continue
#         # Confusing
#         patient_depts = [SYMPTOM_TO_DEPT.get(symptom.lower(), "General Medicine") for symptom in patient.symptoms] # Confusing
#         most_common = Counter(patient_depts).most_common(1)[0][0] # Confusing
#         departments.append(most_common)
#     dept_counts = dict(Counter(department_starts))
#     return StatsOutput(departement_counts=dept_counts)

@app.post("/stats", response_model=StatsOutput)
async def department_stats(patients: List[PatientInput]):
    logger.debug(f"Received /stats request with {len(patients)} patients: {patients}")
    if not patients:
        logger.warning("Empty patient list received")
        raise HTTPException(status_code=400, detail="Patient list cannot be empty")
    
    departments = []
    for patient in patients:
        if not patient.symptoms:
            logger.debug(f"Skipping patient with no symptoms: {patient}")
            continue
        patient_depts = [SYMPTOM_TO_DEPT.get(symptom.lower(), "General Medicine") for symptom in patient.symptoms]
        most_common = Counter(patient_depts).most_common(1)[0][0] if patient_depts else "General Medicine"
        departments.append(most_common)
        logger.debug(f"Patient {patient}: selected department {most_common}")
    
    dept_counts = dict(Counter(departments))
    logger.info(f"Returning department counts: {dept_counts}")
    return StatsOutput(department_counts=dept_counts if dept_counts else {})