import os
import pandas as pd
from groq import Groq
from transformers import pipeline

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def refine_text(text):

    if not text:
        return ""

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": text
            }
        ],
        model="llama-3.3-70b-versatile"
    )

    return response.choices[0].message.content


pipe = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)

def extract_emotions(text):

    preds = pipe(text[:512])

    return {
        p["label"]: p["score"]
        for p in preds[0]
    }
