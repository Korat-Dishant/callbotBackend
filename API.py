from fastapi import FastAPI , Header
from typing import Annotated, List, Union
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from helper import ask_ai , initialize_embeddings , initialize_llm 
# from clearBuffer import clearMemory
import os 
from dotenv import load_dotenv
load_dotenv(override=True)

app = FastAPI(title="Customized Chatbot", description="Customized Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


initialize_llm()
initialize_embeddings()


@app.post("/ask" ,  summary="pass value using header")
async def read_items(user_query:str , user_name:str, user_email:str, date:str):
    return {"response": str(ask_ai(companyName = "callbot", query= user_query , user_name= user_name ,user_email=user_email,date=date )) }



@app.get("/")
async def index():
    return {"message": "Hello World use //extract for api"}


# if __name__ == "__main__":
#     print(ask_ai("what is my name ?"))
