import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from twin.config import DATA, OUT  # noqa: E402
from twin.explain import TinyMLExplainer, top_features  # noqa: E402
from twin.tinyml_model import TinyMLModel  # noqa: E402

model = TinyMLModel(DATA / "fl_global_model_history.json")
op = pd.read_csv(DATA / "digital_twin_input.csv")


background = op[op.ground_truth_label == "NORMAL"].sample(50, random_state=42)
explainer = TinyMLExplainer(model, background)

rows = []
for i, r in op.iterrows():
    e = explainer.explain_row(r, r.model_version, r.predicted_fault)
    row = {"record_id": r.record_id, "site_id": r.site_id, "model_version": r.model_version,
           "predicted_fault": r.predicted_fault, "confidence": round(e["prob"], 4),
           "base_value": round(e["base_value"], 5)}
    row.update({f"shap_{k}": round(v, 5) for k, v in e["values"].items()})
    for n, t in enumerate(top_features(e["values"]), 1):
        row[f"top_feature_{n}"] = t
    row["additivity_error"] = round(e["additivity_error"], 6)
    rows.append(row)
    if i % 500 == 0:
        print(f"  explained {i}/{len(op)}")
shap_df = pd.DataFrame(rows)
shap_df.to_csv(OUT / "shap_values.csv", index=False)

cols = [f"shap_{f}" for f in model.features]
glob = shap_df.groupby("predicted_fault")[cols].apply(lambda g: g.abs().mean())
glob.loc["ALL"] = shap_df[cols].abs().mean()
glob.columns = model.features
glob.round(4).to_csv(OUT / "shap_global.csv")

ax = glob.loc["ALL"].sort_values().plot.barh(figsize=(7, 4), color="#2e7d32")
ax.set_xlabel("mean |SHAP value|  (impact on predicted-fault probability)")
ax.set_title("TinyML fault model - global feature importance")
plt.tight_layout()
plt.savefig(OUT / "shap_global.png", dpi=150)

ref = pd.read_csv(DATA / "member2_shap_values.csv").set_index("record_id").loc[shap_df.record_id]
ours = shap_df.set_index("record_id")
corr = [np.corrcoef(ours.loc[k, cols].astype(float), ref.loc[k, cols].astype(float))[0, 1]
        for k in ours.index]
check = {
    "rows": len(ours),
    "max_additivity_error": float(ours.additivity_error.max()),
    "top1_feature_agreement": round(float(
        (ours.top_feature_1.str.split(" ").str[0] == ref.top_feature_1.str.split(" ").str[0]).mean()), 4),
    "median_row_correlation": round(float(np.nanmedian(corr)), 4),
    "sign_agreement": round(float((np.sign(ours[cols].to_numpy(float)) ==
                                   np.sign(ref[cols].to_numpy(float))).mean()), 4),
}
(OUT / "shap_vs_member2.json").write_text(json.dumps(check, indent=1))

print("\nGlobal importance (mean |SHAP|):")
print(glob.round(3).to_string())
print("\nLocal example:", shap_df.iloc[0][["record_id", "predicted_fault", "confidence",
                                          "top_feature_1", "top_feature_2", "top_feature_3"]].to_dict())
print("\nCheck vs Member 2:", json.dumps(check, indent=1))