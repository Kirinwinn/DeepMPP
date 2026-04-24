import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

df = pd.read_csv("model/AFP_models/AFPwithSolv_model_optical_train_Abs,Emi_20260415_210841/result/prediction.csv")

# 分别评估 train / val / test
for split in ["train", "val", "test"]:
    sub = df[df["set"] == split]
    print(f"\n=== {split.upper()} ({len(sub)} samples) ===")
    for prop in ["Abs", "Emi"]:
        pred = sub[f"{prop}_pred"]
        true = sub[f"{prop}_true"]
        mask = true.notna() & pred.notna()
        pred, true = pred[mask], true[mask]
        r2 = r2_score(true, pred)
        rmse = np.sqrt(mean_squared_error(true, pred))
        mae = mean_absolute_error(true, pred)
        print(f"  {prop}: R²={r2:.4f}, RMSE={rmse:.2f} nm, MAE={mae:.2f} nm")