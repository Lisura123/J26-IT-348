import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402
from twin.config import OUT  # noqa: E402
from twin.recommend import recommend  # noqa: E402

out = pd.read_csv(OUT / "digital_twin_output.csv")
alerts = recommend(out)
alerts.to_csv(OUT / "alerts.csv", index=False)

print(f"{len(out)} readings -> {len(alerts)} alerts\n")
print(alerts.groupby(["rule_id", "physics_cause", "priority", "target_response_h"])
      .size().rename("alerts").to_string())
print("\nTop alert:")
print(alerts.iloc[0][["alert_id", "site_id", "physics_cause", "n_readings", "mean_power_loss_pct",
                      "priority", "recommended_action", "assigned_role", "respond_by_utc"]].to_string())