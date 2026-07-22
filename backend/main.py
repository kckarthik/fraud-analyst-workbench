from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import alerts, chat, score

app = FastAPI(title="Fraud Intel API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(alerts.router)
app.include_router(chat.router)
app.include_router(score.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}
