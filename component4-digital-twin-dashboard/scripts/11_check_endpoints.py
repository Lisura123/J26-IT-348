"""Step 11 check - call every data endpoint as each role."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient  # noqa: E402
from api.main import app  # noqa: E402

client = TestClient(app)


def login(email):
    token = client.post("/auth/login", json={"email": email, "password": "demo123"}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


eng = login("engineer@example.com")

sites = client.get("/sites", headers=eng).json()
print("SITES")
for s in sites:
    print(f"  {s['site_id']}  {s['status']:7s} alerts={s['open_alerts']:2d}  "
          f"latest={s.get('latest', {}).get('physics_cause')}  model={s.get('model', {}).get('model_version')}")

ts = client.get("/sites/SITE-A/timeseries", headers=eng).json()
print(f"\nTIMESERIES SITE-A: {len(ts)} points, first = {ts[0]['timestamp_utc']} "
      f"{ts[0]['power_mw']} mW (expected {ts[0]['expected_power_mw']})")

alerts = client.get("/alerts", headers=eng).json()
print(f"\nALERTS: {len(alerts)}, first = {alerts[0]['alert_id']} {alerts[0]['priority']} {alerts[0]['physics_cause']}")

print("\nALERT DETAIL ALR-0001 by role:")
for email in ["admin@example.com", "engineer@example.com", "manager@example.com"]:
    h = login(email)
    d = client.get("/alerts/ALR-0001", headers=h).json()
    shap_top = d["shap"]["features"][0]["feature"] if d["shap"] else "-"
    direct = client.get(f"/readings/{d['first_record_id']}/shap", headers=h).status_code
    rep = client.get("/metrics", headers=h).status_code
    print(f"  {email.split('@')[0]:9s} shap_visible={d['shap_visible']!s:5s} top={shap_top:12s} "
          f"direct /shap -> {direct}  /metrics -> {rep}")

m = client.get("/metrics", headers=eng).json()
print("\nMETRICS:", m["power"], m["conformal"])