from fastapi import FastAPI
from pydantic import BaseModel
from langchain_agent import ask_question

app = FastAPI()


# Modèle pour la requête
class QuestionRequest(BaseModel):
    question: str


# Endpoint pour question
@app.post("/ask")
async def ask_question_api(request: QuestionRequest):
    try:
        reponse = ask_question(request.question)
        return {"question": request.question, "reponse": reponse}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
