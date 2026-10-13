from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import auth

app = FastAPI(title="Member 4 - Digital Twin API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:4200"],
                   allow_methods=["*"], allow_headers=["*"])


class LoginRequest(BaseModel):
    email: str
    password: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/auth/login")
def login(body: LoginRequest):
    return auth.login(body.email, body.password)


@app.get("/auth/me")
def me(user: dict = Depends(auth.current_user)):
    return auth.public_user(user)


@app.get("/auth/check-shap")
def check_shap(user: dict = Depends(auth.require("view_shap"))):
    return {"ok": True, "role": user["role"]}