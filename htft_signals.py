"""Oranlar arasi bagintilar 1/2 ve 2/1 (devre/mac sonu donusu) icin bilgi tasiyor mu?

Kademeli modeller walk-forward (her sezon yalnizca onceki sezonlarla) egitilir; her kademe
bir oran grubunu ekler. Eklenen grup test log-loss'unu dusuruyorsa sinyal var demektir.
Calistir: python htft_signals.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backtest import DATA

TARGETS = {"1/2": "y12", "2/1": "y21", "donus (1/2 veya 2/1)": "yrev"}
RNG = np.random.default_rng(0)


def load():
    df = pd.read_csv(DATA, dtype={"ft_result": str, "ht_result": str}, parse_dates=["date"]).sort_values("date")
    need = ["spread_home_points", "spread_home_decimal_odds", "spread_away_decimal_odds", "total_over_points"]
    df = df.dropna(subset=need).reset_index(drop=True)
    df["y12"] = (df.htft == "1/2").astype(int)
    df["y21"] = (df.htft == "2/1").astype(int)
    df["yrev"] = df.y12 | df.y21

    p = df.p_ml_home
    df["p2"] = (p - 0.5) ** 2                      # macin dengesi (0 = basa bas)
    df["closeness"] = -(p - 0.5).abs()
    # alt/ust cizgisi: sezonlar arasi skor enflasyonunu ayiklamak icin son 300 macin ortalamasina gore
    df["total_rel"] = df.total_over_points - df.total_over_points.shift(1).rolling(300, min_periods=50).mean()
    df["total_rel"] = df["total_rel"].fillna(0)
    df["spread_ratio"] = df.spread_home_points.abs() / df.total_over_points   # handikap / toplam
    df["tt_home"] = (df.total_over_points - df.spread_home_points) / 2        # takim toplamlari
    df["tt_away"] = (df.total_over_points + df.spread_home_points) / 2
    # mac sonucu orani ile handikap cizgisi uyusmazligi: handikaba gore beklenen olasiliktan sapma
    old = df.season < 2024   # egri yalnizca test oncesi sezonlardan
    fit = np.polyfit(df.spread_home_points[old], np.log(p / (1 - p))[old], 3)
    df["ml_vs_spread"] = np.log(p / (1 - p)) - np.polyval(fit, df.spread_home_points)
    df["juice_spread"] = df.p_spread_home - 0.5    # handikap oranlarindaki egim
    df["juice_total"] = df.p_over - 0.5            # alt/ust oranlarindaki egim
    df["margin_ml"] = 1 / df.money_home_decimal_odds + 1 / df.money_away_decimal_odds - 1
    return df


STAGES = [
    ("A  yalniz mac sonucu orani", ["p_ml_home", "p2"]),
    ("B  + alt/ust cizgisi", ["total_over_points", "total_rel"]),
    ("C  + handikap cizgisi, takim toplamlari", ["spread_home_points", "spread_ratio", "tt_home", "tt_away"]),
    ("D  + MS-handikap uyusmazligi", ["ml_vs_spread", "margin_ml"]),
    ("E  + oran egimleri (handikap, alt/ust)", ["juice_spread", "juice_total"]),
]
PUBLIC = ["money_home_stake_percentage", "money_home_wager_percentage", "spread_home_stake_percentage",
          "spread_home_wager_percentage", "total_over_stake_percentage", "total_over_wager_percentage"]


def walk(df, cols, target, model="logit", first_test=2024):
    pred = pd.Series(np.nan, index=df.index)
    for s in sorted(x for x in df.season.unique() if x >= first_test):
        tr, te = df[df.season < s], df[df.season == s]
        if model == "logit":
            m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        else:
            m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03, max_iter=200,
                                               min_samples_leaf=80, l2_regularization=1.0, random_state=0)
        m.fit(tr[cols], tr[target])
        pred[te.index] = m.predict_proba(te[cols])[:, 1]
    return pred.dropna()


def ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def compare(base, new, y, n=2000):
    """Log-loss farki (new - base) ve bootstrap ile %95 araligi; negatif = new daha iyi."""
    d = (ll(new, y) - ll(base, y)).to_numpy()
    boots = [d[RNG.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def run(df, label, stages):
    for name, target in TARGETS.items():
        print(f"\n--- {label} | hedef: {name} ---")
        cols, base = [], None
        for stage, add in stages:
            cols = cols + add
            p = walk(df, cols, target)
            y = df.loc[p.index, target]
            line = f"{stage:42s} log-loss={ll(p, y).mean():.5f}  AUC={roc_auc_score(y, p):.3f}"
            if base is None:
                base = p
                const = pd.Series(np.nan, index=p.index)
                for s in df.loc[p.index, "season"].unique():
                    const[df.index[(df.season == s) & df.index.isin(p.index)]] = df[df.season < s][target].mean()
                line += f"   (sabit frekans log-loss={ll(const, y).mean():.5f})"
            else:
                d, lo, hi = compare(base, p, y)
                line += f"   A'ya gore fark={d:+.5f} [{lo:+.5f}, {hi:+.5f}]"
            print(line)
        p = walk(df, cols, target, model="gbm")
        d, lo, hi = compare(base, p, y)
        print(f"{'GBM (tum ozellikler, dogrusal olmayan)':42s} log-loss={ll(p, y).mean():.5f}  "
              f"AUC={roc_auc_score(y, p):.3f}   A'ya gore fark={d:+.5f} [{lo:+.5f}, {hi:+.5f}]")


def tables(df):
    """Mac sonucu orani sabitken diger oranlara gore 1/2 ve 2/1 sikligi."""
    df = df.copy()
    df["fav"] = pd.cut(df.p_ml_home, [0, .3, .45, .55, .7, 1], labels=["dep. net fav", "dep. hafif fav", "basa bas", "ev hafif fav", "ev net fav"])
    for col, title in [("total_rel", "alt/ust cizgisi (ortalamaya gore)"), ("ml_vs_spread", "MS orani handikaba gore (yuksek = ev MS'de daha favori)"),
                       ("juice_spread", "handikap oran egimi (yuksek = ev tarafi pahali)"), ("juice_total", "alt/ust oran egimi (yuksek = ust pahali)")]:
        if df[col].nunique() < 60:   # oran egimleri az sayida deger aliyor: isarete gore ayir
            df["g"] = pd.cut(df[col], [-1, -1e-9, 1e-9, 1], labels=["dusuk", "orta", "yuksek"])
        else:
            df["g"] = pd.qcut(df[col], 3, labels=["dusuk", "orta", "yuksek"])
        t = df.groupby(["fav", "g"], observed=True).agg(mac=("y12", "size"), r12=("y12", "mean"), r21=("y21", "mean")).round(3)
        print(f"\n{title}\n{t.unstack('g').to_string()}")


if __name__ == "__main__":
    d = load()
    print(f"{len(d)} mac; 1/2={d.y12.mean():.3f} 2/1={d.y21.mean():.3f}")
    run(d, "TUM MACLAR", STAGES)
    pub = d.dropna(subset=PUBLIC)
    run(pub, f"BAHIS/PARA YUZDESI OLAN MACLAR ({len(pub)})", [(STAGES[0][0], STAGES[0][1]), ("F  + bahis ve para yuzdeleri", PUBLIC)])
    tables(d)
