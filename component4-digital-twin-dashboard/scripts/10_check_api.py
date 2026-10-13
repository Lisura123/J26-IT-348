import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient  # noqa: E402
from api.main import app  # noqa: E402

client = TestClient(app)
print("health:", client.get("/health").json())
print("wrong password:", client.post("/auth/login", json={"email": "admin@example.com", "password": "x"}).status_code)

for email in ["admin@example.com", "engineer@example.com", "manager@example.com"]:
    r = client.post("/auth/login", json={"email": email, "password": "demo123"}).json()
    headers = {"Authorization": f"Bearer {r['access_token']}"}
    me = client.get("/auth/me", headers=headers).json()
    shap = client.get("/auth/check-shap", headers=headers)
    print(f"{me['role']:9s} permissions={len(me['permissions']):2d}  view_shap -> HTTP {shap.status_code}")

print("no token:", client.get("/auth/me").status_code)