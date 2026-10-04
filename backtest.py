"""Kapanis oranlarina gore "benzer mac" (kNN) tahmini ve ileriye donuk (walk-forward) backtest.

Her test sezonu yalnizca kendinden ONCEKI sezonlarla tahmin edilir; gelecege bakma yok.
Calistir: python backtest.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data" / "nba_betmgm_closing.csv"
FEATURES = ["p_ml_home", "total_over_points"]
HTFT = ["1/1", "1/2", "2/1", "2/2", "X/1", "X/2"]
K = 500          # komsu sayisi
EDGE = 0.05      # bahis esigi: tahmini olasilik * oran > 1 + EDGE


def load():
    df = pd.read_csv(DATA, dtype={"ft_result": str, "ht_result": str}, parse_dates=["date"])
    df["home_win"] = (df["ft_result"] == "1").astype(float)
    df["over"] = (df["total_result"] == "over").astype(float)
    df["push"] = df["total_result"] == "push"
    for c in HTFT:
        df[f"y_{c}"] = (df["htft"] == c).astype(float)
    return df


def knn_predict(train, test, k=K):
    """Her test maci icin, oranlari en benzer k gecmis macin sonuc frekanslari."""
    mu, sd = train[FEATURES].mean(), train[FEATURES].std()
    a = ((train[FEATURES] - mu) / sd).to_numpy()
    b = ((test[FEATURES] - mu) / sd).to_numpy()
    targets = ["home_win", "over"] + [f"y_{c}" for c in HTFT]
    y = train[targets].to_numpy()
    k = min(k, len(a))
    out = np.empty((len(b), len(targets)))
    for i, row in enumerate(b):
        idx = np.argpartition(((a - row) ** 2).sum(1), k - 1)[:k]
        out[i] = y[idx].mean(0)
    cols = ["pred_home", "pred_over"] + [f"pred_{c}" for c in HTFT]
    return pd.DataFrame(out, columns=cols, index=test.index)


def walk_forward(df, first_test=2024):
    parts = []
    for season in sorted(s for s in df["season"].unique() if s >= first_test):
        train, test = df[df["season"] < season], df[df["season"] == season]
        parts.append(test.join(knn_predict(train, test)))
    return pd.concat(parts)


def roi(profit):
    n = len(profit)
    return f"n={n:5d}  kar={profit.sum():8.1f} birim  ROI={profit.mean() * 100 if n else 0:6.2f}%"


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p)).sum(1).mean()


def report(r, df):
    print(f"Test: {len(r)} mac, sezonlar {sorted(r.season.unique())}, k={K}\n")

    print("== MAC SONUCU (kapanis oraniyla 1 birim) ==")
    win_h, win_a = r.home_win == 1, r.home_win == 0
    pl_home = np.where(win_h, r.money_home_decimal_odds - 1, -1.0)
    pl_away = np.where(win_a, r.money_away_decimal_odds - 1, -1.0)
    fav_home = r.p_ml_home >= 0.5
    print("hep ev sahibi     ", roi(pd.Series(pl_home)))
    print("hep deplasman     ", roi(pd.Series(pl_away)))
    print("hep favori        ", roi(pd.Series(np.where(fav_home, pl_home, pl_away))))
    print("hep surpriz       ", roi(pd.Series(np.where(fav_home, pl_away, pl_home))))
    bh = r.pred_home * r.money_home_decimal_odds > 1 + EDGE
    ba = (1 - r.pred_home) * r.money_away_decimal_odds > 1 + EDGE
    print(f"model (avantaj>{EDGE:.0%})", roi(pd.Series(np.concatenate([pl_home[bh], pl_away[ba]]))))

    print("\n== ALT / UST ==")
    np_ = ~r.push
    pl_over = np.where(r.push, 0.0, np.where(r.over == 1, r.total_over_decimal_odds - 1, -1.0))
    pl_under = np.where(r.push, 0.0, np.where(r.over == 0, r.total_under_decimal_odds - 1, -1.0))
    print("hep ust           ", roi(pd.Series(pl_over)))
    print("hep alt           ", roi(pd.Series(pl_under)))
    bo = r.pred_over * r.total_over_decimal_odds > 1 + EDGE
    bu = (1 - r.pred_over) * r.total_under_decimal_odds > 1 + EDGE
    print(f"model (avantaj>{EDGE:.0%})", roi(pd.Series(np.concatenate([pl_over[bo], pl_under[bu]]))))
    print(f"model ust tahmini ile gercek korelasyon: {np.corrcoef(r.pred_over[np_], r.over[np_])[0, 1]:.3f}")

    print("\n== IY/MS (oran verisi yok; yalnizca tahmin kalitesi) ==")
    y = r[[f"y_{c}" for c in HTFT]].to_numpy()
    p = r[[f"pred_{c}" for c in HTFT]].to_numpy()
    base = np.vstack([df[df.season < s][[f"y_{c}" for c in HTFT]].mean().to_numpy() for s in r.season])
    print(f"log-loss  model={logloss(p, y):.4f}  sabit frekans={logloss(base, y):.4f}  (dusuk = iyi)")
    for c in ("1/2", "2/1"):
        q = pd.qcut(r[f"pred_{c}"], 5, duplicates="drop")
        t = r.groupby(q, observed=True).agg(mac=(f"y_{c}", "size"), tahmin=(f"pred_{c}", "mean"), gercek=(f"y_{c}", "mean"))
        t["adil_oran"] = 1 / t["tahmin"]
        print(f"\n{c} kalibrasyonu (tahmine gore 5 dilim):")
        print(t.round(3).to_string())


if __name__ == "__main__":
    data = load()
    report(walk_forward(data), data)
