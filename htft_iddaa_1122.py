"""1/1 ve 2/2 oranlari 1/2 ve 2/1 icin isaret veriyor mu? (gercek IDDAA oranlari, Mackolik arsivi)

Taban: Iddaa'nin kendi marjsiz 1/2 (2/1) olasiligi -- canlida elimizde olacak tek bilgi bu.
Her aday alan icin: lojistik  y ~ logit(q_hedef) + alan   ile  y ~ logit(q_hedef)  karsilastirilir (olabilirlik orani).
  kesif     : 2021/22 - 2023/24      dogrulama : 2024/25 - 2025/26
  Benjamini-Hochberg kesifte; kesifte anlamli olan dogrulamada AYNI yonde ve p < 0.05 olmali.
Ayrica her alanin besli dilimlerinde 1/2 ve 2/1 getirisi (butun veri) gosterilir.

Calistir: python htft_iddaa_1122.py      (once htft_iddaa.py -> data/nba_iddaa.csv)
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression

from htft_deep import REG, ROOT, bh, h
from htft_iddaa import implied

lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
DISCOVERY, VALIDATION = (2022, 2023, 2024), (2025, 2026)


def load():
    df = implied(pd.read_csv(ROOT / "nba_iddaa.csv", dtype={"match_id": str}, parse_dates=["date"]))
    df = df[df[[f"iy_{c}" for c in ("1/1", "2/2", "1/2", "2/1")]].notna().all(axis=1)].reset_index(drop=True)
    q = {c: df[f"q_{c}"] for c in REG}
    ms1 = q["1/1"] + q["X/1"] + q["2/1"]                    # IY/MS'nin ima ettigi ev kazanir (normal sure)
    ms2 = q["1/2"] + q["X/2"] + q["2/2"]
    ht1 = q["1/1"] + q["1/X"] + q["1/2"]                    # IY/MS'nin ima ettigi ev devrede onde
    ht2 = q["2/1"] + q["2/X"] + q["2/2"]
    F = pd.DataFrame(index=df.index)
    F["oran_1/1"], F["oran_2/2"] = df["iy_1/1"], df["iy_2/2"]
    F["q_1/1"], F["q_2/2"] = q["1/1"], q["2/2"]
    F["log_11_bolu_22"] = np.log(df["iy_1/1"] / df["iy_2/2"])
    F["log_11_carpi_22"] = np.log(df["iy_1/1"] * df["iy_2/2"])
    F["oran_11_arti_22"] = df["iy_1/1"] + df["iy_2/2"]
    F["P(devre1|MS1)"] = q["1/1"] / ms1                       # ev kazanirsa devrede de onde olma olasiligi
    F["P(devre2|MS2)"] = q["2/2"] / ms2
    F["P(MS1|devre1)"] = q["1/1"] / ht1                       # ev devrede ondeyse kazanma olasiligi
    F["P(MS2|devre2)"] = q["2/2"] / ht2
    F["11+22_toplam_q"] = q["1/1"] + q["2/2"]                 # "donus olmaz" toplam olasiligi
    F["11_22_marj_payi"] = (1 / df["iy_1/1"] + 1 / df["iy_2/2"]) / (1 / df[[f"iy_{c}" for c in REG]]).sum(axis=1)
    if {"iy1x2_1", "iy1x2_2", "iy1x2_X"} <= set(df.columns):  # ayri "1. yari sonucu" marketi ile tutarsizlik
        fh = 1 / df[["iy1x2_1", "iy1x2_X", "iy1x2_2"]]
        fh = fh.div(fh.sum(axis=1), axis=0)
        F["devre1_IYMS_eksi_IYmarketi"] = ht1 - fh["iy1x2_1"]
        F["devre2_IYMS_eksi_IYmarketi"] = ht2 - fh["iy1x2_2"]
    if {"ms1", "ms2"} <= set(df.columns):                       # MS marketi ile tutarsizlik
        p1 = (1 / df["ms1"]) / (1 / df["ms1"] + 1 / df["ms2"])
        F["MS1_IYMS_eksi_MSmarketi"] = ms1 / (ms1 + ms2) - p1
    return df, F


def lr(d, y, base, x):
    """y ~ base + x  vs  y ~ base; (LR, x katsayisi isareti)."""
    b = d[[base]].values
    bx = np.column_stack([b, (d[x] - d[x].mean()) / d[x].std()])
    m0 = LogisticRegression(C=1e6, max_iter=1000).fit(b, d[y])
    m1 = LogisticRegression(C=1e6, max_iter=1000).fit(bx, d[y])
    ll = lambda m, X: np.sum(np.log(np.where(d[y] == 1, m.predict_proba(X)[:, 1], m.predict_proba(X)[:, 0])))
    return 2 * (ll(m1, bx) - ll(m0, b)), np.sign(m1.coef_[0][-1])


def main():
    pd.set_option("display.width", 250)
    df, F = load()
    print(f"{len(df)} mac, {F.shape[1]} aday alan")
    rows = []
    for c in ("1/2", "2/1"):
        y, base = f"y{c}", f"lq{c}"
        d = pd.concat([df[["season", "htft_reg", f"q_{c}", f"iy_{c}"]], F], axis=1)
        d[y], d[base] = (d.htft_reg == c).astype(int), lg(d[f"q_{c}"])
        for x in F.columns:
            dd = d.dropna(subset=[x])
            if dd[x].std() == 0:
                continue
            k, v = dd[dd.season.isin(DISCOVERY)], dd[dd.season.isin(VALIDATION)]
            l1, s1 = lr(k, y, base, x)
            l2, s2 = lr(v, y, base, x)
            rows.append({"hedef": c, "alan": x, "kesif_p": chi2.sf(max(l1, 0), 1), "kesif_yon": s1,
                         "dogrulama_p": chi2.sf(max(l2, 0), 1), "dogrulama_yon": s2})
    t = pd.DataFrame(rows)
    t["kesif_BH"] = bh(t.kesif_p)
    t["tuttu"] = t.kesif_BH & (t.dogrulama_p < 0.05) & (t.kesif_yon == t.dogrulama_yon)
    h("1/1 ve 2/2 KAYNAKLI ALANLAR: Iddaa'nin kendi 1/2 (2/1) olasiligina EK bilgi veriyor mu?")
    print(t.sort_values("kesif_p").round(4).to_string(index=False))
    print(f"\n{len(t)} test. Kesifte (BH sonrasi) anlamli: {t.kesif_BH.sum()}, duzeltmesiz kesif p<0.05: {(t.kesif_p < 0.05).sum()} "
          f"(sansla ~{0.05 * len(t):.1f}). Dogrulamada ayni yonde tekrarlayan: {t.tuttu.sum()}")
    print(f"Kesif ve dogrulama yonlerinin uyusma orani: {(t.kesif_yon == t.dogrulama_yon).mean():.2f} (sans 0.50)")

    h("BESLI DILIMLER: her alanin dusukten yuksege 5 diliminde 1/2 ve 2/1 getirisi (butun veri, 1 birim)")
    for c in ("1/2", "2/1"):
        pl = np.where(df.htft_reg == c, df[f"iy_{c}"] - 1, -1.0)
        out = {}
        for x in F.columns:
            v = F[x].dropna()
            if v.nunique() < 5:
                continue
            g = pd.qcut(v.rank(method="first"), 5, labels=False)
            out[x] = pd.Series(pl[v.index]).groupby(g.values).mean()
        print(f"\n{c} getirisi (sutun 0 = alanin en dusuk %20'si, 4 = en yuksek %20'si):")
        print(pd.DataFrame(out).T.round(3).to_string())


if __name__ == "__main__":
    main()
