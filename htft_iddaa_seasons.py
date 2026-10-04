"""Gercek IDDAA oranlariyla sezon dinamigi: 1/2 ve 2/1 fiyatlamasi sezona gore degisiyor mu, buna uyum saglamak kazandirir mi?

  S1  Her sezon, her oran diliminde: Iddaa'nin ima ettigi (marjsiz) olasilik vs gercek siklik ve getiri.
      Bir sezonun dilim getirileri bir sonraki sezonun dilim getirilerini ongoruyor mu (korelasyon)?
  S2  Uyarlanan model (walk-forward, her gun yeniden egitilir, gelecegi gormez):
        P(sonuc) = lojistik( logit(Iddaa marjsiz olasilik), mu, mu^2 )
      Dort egitim penceresi: butun gecmis / yalniz onceki sezon / yalniz bu sezonun gecmisi / son 300 mac.
      Olcu: log-loss (Iddaa'nin kendi olasiligina gore iyilesme) ve "model olasiligi x oran > 1 + esik" bahsinin getirisi.
  S3  Sezon ici: ilk yarida kar eden oran dilimleri ikinci yarida da kar ediyor mu?

Calistir: python htft_iddaa_seasons.py      (once htft_iddaa.py -> data/nba_iddaa.csv)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from htft_deep import ROOT, h
from htft_iddaa import implied

TARGETS = ("1/2", "2/1")
BINS = {"1/2": [0, 5, 6, 7, 8.5, 11, 15, 1000], "2/1": [0, 4.5, 5.5, 6.5, 8, 11, 1000]}
WINDOWS = ("butun_gecmis", "onceki_sezon", "bu_sezon", "son_300")
EDGES = (0.0, 0.05, 0.10, 0.20)
MIN_TRAIN = 150
lg = lambda p: np.log(p / (1 - p))


def load():
    df = implied(pd.read_csv(ROOT / "nba_iddaa.csv", dtype={"match_id": str}, parse_dates=["date"]))
    df = df[df["iy_1/2"].notna() & df["iy_2/1"].notna() & df["mu"].notna()].sort_values("date").reset_index(drop=True)
    for c in TARGETS:
        df[f"y{c}"] = (df.htft_reg == c).astype(int)
        df[f"pl{c}"] = np.where(df[f"y{c}"] == 1, df[f"iy_{c}"] - 1, -1.0)
    return df


def roi_t(pl):
    """Ortalama getiri ve t istatistigi (0'dan farki)."""
    pl = np.asarray(pl)
    if len(pl) < 2:
        return (pl.mean() if len(pl) else np.nan), np.nan
    return pl.mean(), pl.mean() / (pl.std(ddof=1) / np.sqrt(len(pl)))


# ------------------------------------------------------------------ S1
def s1(df):
    h("S1. HER SEZON, HER ORAN DILIMI: Iddaa marjsiz olasilik / gercek siklik / getiri")
    for c in TARGETS:
        df["bin"] = pd.cut(df[f"iy_{c}"], BINS[c])
        g = df.groupby(["bin", "season"], observed=True)
        t = g.agg(mac=(f"y{c}", "size"), iddaa=(f"q_{c}", "mean"), gercek=(f"y{c}", "mean"), getiri=(f"pl{c}", "mean"))
        print(f"\n{c} -- getiri (satir: oran dilimi, sutun: sezon; parantez: mac sayisi):")
        piv = t.getiri.unstack().round(2).astype(str) + " (" + t.mac.unstack().fillna(0).astype(int).astype(str) + ")"
        print(piv.to_string())
        print(f"\n{c} -- gercek siklik eksi Iddaa marjsiz olasilik (+ ise Iddaa olasiligi dusuk veriyor):")
        print((t.gercek - t.iddaa).unstack().round(3).to_string())
        r = t.getiri.unstack()
        cors = [r[s].corr(r[s + 1]) for s in r.columns[:-1]]
        print(f"Ardisik sezonlarin dilim getirileri arasi korelasyon: "
              + ", ".join(f"{s}->{s + 1}: {v:+.2f}" for s, v in zip(r.columns[:-1], cors)) + f"  (ortalama {np.mean(cors):+.2f})")
        best = r.idxmax()
        nxt = [(s, best[s], r.loc[best[s], s + 1]) for s in r.columns[:-1]]
        print("Her sezonun en iyi dilimi bir sonraki sezonda: "
              + "; ".join(f"{s} en iyi {b} ({r.loc[b, s]:+.2f}) -> {s + 1}: {v:+.2f}" for s, b, v in nxt))


# ------------------------------------------------------------------ S2
def features(d, c):
    mu = d["mu"].values
    return np.column_stack([lg(d[f"q_{c}"].clip(1e-4, 1 - 1e-4)), mu / 10, (mu / 10) ** 2])


def walk_forward(df, c, window):
    p = pd.Series(np.nan, index=df.index)
    for day, te in df.groupby("date"):
        past = df[df.date < day]
        s = te.season.iloc[0]
        if window == "onceki_sezon":
            past = past[past.season == s - 1]
        elif window == "bu_sezon":
            past = past[past.season == s]
        elif window == "son_300":
            past = past.tail(300)
        if len(past) < MIN_TRAIN or past[f"y{c}"].nunique() < 2:
            continue
        m = LogisticRegression(C=1.0, max_iter=500).fit(features(past, c), past[f"y{c}"])
        p[te.index] = m.predict_proba(features(te, c))[:, 1]
    return p


def logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def s2(df):
    h("S2. SEZONA UYARLANAN MODEL (walk-forward, gunluk yeniden egitim)")
    test = df.season >= df.season.min() + 1                                     # ilk sezon yalniz egitim
    for c in TARGETS:
        print(f"\n===== {c} =====")
        preds = {w: walk_forward(df, c, w) for w in WINDOWS}
        common = test & np.logical_and.reduce([preds[w].notna() for w in WINDOWS])
        d = df[common]
        base = logloss(d[f"y{c}"], d[f"q_{c}"])
        rows = []
        for w in WINDOWS:
            diff = logloss(d[f"y{c}"], preds[w][common]) - base                # < 0 ise model Iddaa'dan iyi
            rows.append({"pencere": w, "mac": len(d), "logloss_farki_x1000": 1000 * diff.mean(),
                         "t": diff.mean() / (diff.std(ddof=1) / np.sqrt(len(d))),
                         **{f"{s}": 1000 * diff[d.season == s].mean() for s in sorted(d.season.unique())}})
        print("Log-loss farki (model - Iddaa marjsiz olasiligi) x1000; negatif = model daha iyi tahmin ediyor:")
        print(pd.DataFrame(rows).round(2).to_string(index=False))

        rows = []
        for w in WINDOWS:
            for e in EDGES:
                ev = preds[w] * df[f"iy_{c}"] - 1
                msk = common & (ev > e)
                m, t = roi_t(df.loc[msk, f"pl{c}"])
                rec = {"pencere": w, "esik": e, "bahis": int(msk.sum()), "isabet": df.loc[msk, f"y{c}"].mean(),
                       "ort_oran": df.loc[msk, f"iy_{c}"].mean(), "getiri": m, "t": t}
                for s in sorted(df[common].season.unique()):
                    ms = msk & (df.season == s)
                    rec[f"{s}"] = f"{df.loc[ms, f'pl{c}'].mean():+.2f}/{int(ms.sum())}" if ms.any() else "-"
                rows.append(rec)
        print("\nBahis: model olasiligi x Iddaa orani > 1 + esik. Sezon sutunlari: getiri/bahis sayisi. "
              "|t| > 2 olmadikca sans disi sayilmaz:")
        print(pd.DataFrame(rows).round(3).to_string(index=False))


# ------------------------------------------------------------------ S3
def s3(df):
    h("S3. SEZON ICI: ilk yarida kar eden dilim ikinci yarida da kar ediyor mu?")
    rows = []
    for s in sorted(df.season.unique()):
        ds = df[df.season == s]
        mid = ds.date.iloc[len(ds) // 2]
        first, second = ds[ds.date < mid], ds[ds.date >= mid]
        for c in TARGETS:
            for side, sm in (("hepsi", None), ("ev_favori", True), ("dep_favori", False)):
                for b in pd.cut(ds[f"iy_{c}"], BINS[c]).cat.categories:
                    f1 = first[first[f"iy_{c}"].between(b.left, b.right, inclusive="right")]
                    f2 = second[second[f"iy_{c}"].between(b.left, b.right, inclusive="right")]
                    if sm is not None:
                        f1, f2 = f1[(f1.mu > 0) == sm], f2[(f2.mu > 0) == sm]
                    if len(f1) < 30 or len(f2) < 30:
                        continue
                    rows.append({"sezon": s, "secenek": c, "taraf": side, "dilim": str(b), "n1": len(f1),
                                 "ilk_yari": f1[f"pl{c}"].mean(), "n2": len(f2), "ikinci_yari": f2[f"pl{c}"].mean()})
    t = pd.DataFrame(rows)
    win = t[t.ilk_yari > 0]
    print(f"{len(t)} (sezon x secenek x taraf x dilim) hucresi. Ilk yarida artida olan: {len(win)}.")
    print(f"Bunlarin ikinci yarida da artida kalan: {(win.ikinci_yari > 0).sum()} / {len(win)}; "
          f"ikinci yaridaki ortalama getirileri: {win.ikinci_yari.mean():+.3f}  "
          f"(ilk yarida eksi olanlarin ikinci yari ortalamasi: {t[t.ilk_yari <= 0].ikinci_yari.mean():+.3f})")
    print(f"Ilk yari ve ikinci yari getirileri arasi korelasyon: {t.ilk_yari.corr(t.ikinci_yari):+.3f}")
    print("\nIlk yarida en iyi 15 hucre ve ikinci yaridaki sonuclari:")
    print(t.sort_values("ilk_yari", ascending=False).head(15).round(3).to_string(index=False))


def main():
    pd.set_option("display.width", 250)
    df = load()
    print(f"{len(df)} mac, sezonlar {sorted(df.season.unique())}")
    s1(df)
    s3(df)
    s2(df)


if __name__ == "__main__":
    main()
