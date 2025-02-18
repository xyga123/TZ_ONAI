import os
import logging
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

load_dotenv()

db_url = os.getenv("DATABASE_URL")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
llm_api_url = os.getenv(
    "LLM_API_URL")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wh")

app = FastAPI()

engine = create_engine(db_url, connect_args={
                       "check_same_thread": False} if "sqlite" in db_url else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    callback_url = Column(String, nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)


Base.metadata.create_all(bind=engine)


class WebhookRequest(BaseModel):
    message: str
    callback_url: str


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/webhook")
async def webhook_endpoint(request_data: WebhookRequest, db: Session = Depends(get_db)):
    try:
        save_message(db, request_data.callback_url,
                     "user", request_data.message)

        history = get_history(db, request_data.callback_url)

        response_text = await call_llm(history)

        save_message(db, request_data.callback_url, "assistant", response_text)

        await send_callback(request_data.callback_url, response_text)

        return {"status": "success", "response": response_text}

    except Exception as e:
        logger.error(f"Ошибка: {e}")
        raise HTTPException(status_code=500, detail="Ошибка обработки запроса")


def save_message(db: Session, callback_url: str, role: str, content: str):
    message = Message(callback_url=callback_url,
                      role=role, content=content)
    db.add(message)
    db.commit()


def get_history(db: Session, callback_url: str):
    return db.query(Message).filter(Message.callback_url == callback_url).all()


async def call_llm(history: list) -> str:
    headers = {"Content-Type": "application/json"}
    if openrouter_api_key:
        headers["Authorization"] = f"Bearer {openrouter_api_key}"

    payload = {"model": "gpt-3.5-turbo",
               "messages": [{"role": msg.role, "content": msg.content} for msg in history]}

    async with httpx.AsyncClient() as client:
        response = await client.post(llm_api_url, json=payload, headers=headers, timeout=30)
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code, detail="Ошибка LLM API")

        data = response.json()
        return data["choices"][0]["message"]["content"]


async def send_callback(callback_url: str, response_text: str):
    payload = {"response": response_text}
    async with httpx.AsyncClient() as client:
        response = await client.post(callback_url, json=payload, timeout=15)
        logger.info(
            f"Callback отправлен на {callback_url}, статус: {response.status_code}")


# МЕТОД ЗАГЛУШКА
@app.post("/callback")
async def handle_callback(response: dict):
    print(f"Получен callback: {response}")
    return {"status": "received"}
