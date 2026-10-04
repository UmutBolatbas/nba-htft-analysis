"""Sezon sezon 1/2 ve 2/1: "her sezon sistem degisiyor" fikrinin testleri.

  S1  Sezonlarin piyasa yapisi (barem, handikap, marj) ve 1/2-2/1 oranlari; handikaptan beklenene gore sapma (z)
      ve sezonlar arasi homojenlik ki-kare testi
  S2  Brown modeli parametreleri her sezon ayri (bootstrap %95 guven araligi)
  S3  Sezon icinde ay ay ve playoff: gercek vs beklenen
  S4  Hangi egitim penceresi daha iyi tahmin eder? (tum gecmis / yalniz onceki sezon / gecmis + bu sezon /
      yalniz bu sezon) -- sistem her sezon degisiyorsa yakin pencereler kazanmali
  S5  Sezon ici kesif/dogrulama: her sezonun ILK yarisinda bulunan oran alani etkileri AYNI sezonun ikinci
      yarisinda tekrarliyor mu? Ayrica bir sezonun etkileri sonraki sezona tasiniyor mu?

Calistir: python htft_seasons.py
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from htft_deep import CANDS, OT6, REG, add_features, bh, bm_probs, fit_bm, h, load, logloss, lr_test, pct

RNG = np.random.default_rng(0)
MONTHS = {10: "Eki", 11: "Kas", 12: "Ara", 1: "Oca", 2: "Sub", 3: "Mar", 4: "Nis", 5: "May", 6: "Haz"}


def observed_vs_expected(g, cols=("1/2", "2/1")):
    out = {"mac": len(g)}
    for c in cols:
        o, e = (g.htft_reg == c).sum(), g[f"bm_{c}"].sum()
        sd = np.sqrt((g[f"bm_{c}"] * (1 - g[f"bm_{c}"])).sum())
        out[f"{c} gercek"], out[f"{c} beklenen"], out[f"{c} z"] = o / len(g), e / len(g), (o - e) / sd
    return pd.Series(out)


def main():
    pd.set_option("display.width", 220)
    df = load()
    df["half"] = df.groupby("season").date.rank(method="first") <= df.groupby("season").date.transform("size") / 2
    p_all = fit_bm(df)
    df = df.join(bm_probs(df.mu, p_all).add_prefix("bm_"))
    seasons = sorted(df.season.unique())
    print("Sezon etiketi bitis yili: 2022 = 2021-22. 2026 sezonu Subat 2026'da bitiyor (eksik).")

    # ------------------------------------------------------------ S1
    h("S1. SEZONLARIN PIYASA YAPISI VE 1/2-2/1 ORANLARI (normal sure)")
    m = df.groupby("season").agg(
        mac=("htft", "size"), ort_barem=("total_over_points", "mean"), ort_puan=("total_points", "mean"),
        ort_mutlak_handikap=("fav_spread", "mean"),
        ms_marji=("money_home_decimal_odds", lambda s: (1 / s + 1 / df.loc[s.index, "money_away_decimal_odds"] - 1).mean()),
        uzatma=("D", lambda s: (s == 0).mean()), ev_kazanma=("F", lambda s: (s > 0).mean()))
    print(m.round(3).to_string())
    t = df.groupby("season").apply(observed_vs_expected, include_groups=False)
    print("\nGercek vs handikaptan beklenen (z: |z|>2 kabaca anlamli; 5 sezonda 1 tane sansla cikabilir):")
    print(t.round(4).to_string())
    for c in ("1/2", "2/1"):
        stat = (t[f"{c} z"] ** 2).sum()
        print(f"  {c}: sezonlar arasi homojenlik ki-kare = {stat:.2f}, sd = {len(t)}, p = {chi2.sf(stat, len(t)):.3f}")
    tab = pd.crosstab(df.season, df.htft_reg)[REG]
    tab = tab.loc[:, tab.sum() > 0]
    e = np.outer(tab.sum(1), tab.sum(0)) / tab.values.sum()
    stat = ((tab.values - e) ** 2 / e).sum()
    dof = (tab.shape[0] - 1) * (tab.shape[1] - 1)
    print(f"  9 secenegin tamami (sezon x IY/MS tablosu): ki-kare = {stat:.1f}, sd = {dof}, p = {chi2.sf(stat, dof):.3f}")

    # ------------------------------------------------------------ S2
    h("S2. BROWN MODELI PARAMETRELERI SEZON SEZON (parantez: bootstrap %95 araligi)")
    keys = {"a": "ilk yari payi", "sh": "devre sigma", "ss": "2. yari sigma", "sd": "mac sigma",
            "rho_halves": "yari korelasyonu", "p_ot_home": "uzatma ev"}
    rows = {}
    for s in seasons + ["hepsi"]:
        g = df if s == "hepsi" else df[df.season == s]
        est = fit_bm(g)
        boot = pd.DataFrame([fit_bm(g.sample(len(g), replace=True, random_state=RNG.integers(1e9))) for _ in range(200)])
        rows[s] = {v: f"{est[k]:6.3f} ({boot[k].quantile(.025):6.3f},{boot[k].quantile(.975):6.3f})" for k, v in keys.items()}
    print(pd.DataFrame(rows).T.to_string())
    print("Araliklar ust uste biniyorsa o parametre sezonlar arasinda istatistiksel olarak ayirt edilemiyor.")

    # ------------------------------------------------------------ S3
    h("S3. SEZON ICINDE: AY AY VE PLAYOFF")
    df["ay"] = df.date.dt.month.map(MONTHS)
    df["donem"] = np.where(df.playoffs, "playoff", df.ay)
    order = list(MONTHS.values())[:7] + ["playoff"]
    t = df.groupby("donem").apply(observed_vs_expected, include_groups=False).reindex(order).dropna()
    print(t.round(4).to_string())
    for c in ("1/2", "2/1"):
        stat = (t[f"{c} z"] ** 2).sum()
        print(f"  {c}: donemler arasi ki-kare = {stat:.2f}, sd = {len(t)}, p = {chi2.sf(stat, len(t)):.3f}")
    t = df.groupby(["season", "half"]).apply(observed_vs_expected, include_groups=False)
    t.index = t.index.set_levels(["2. yari", "1. yari"], level=1)
    print("\nHer sezonun ilk ve ikinci yarisi:")
    print(t.round(4).to_string())

    # ------------------------------------------------------------ S4
    h("S4. HANGI EGITIM PENCERESI DAHA IYI? (her ay yeniden egitim, test 2023-2026, normal sure 9 secenek log-loss)")
    df["ym"] = df.date.dt.to_period("M")
    res = []
    for s in seasons[1:]:
        for ym in sorted(df.ym[df.season == s].unique()):
            te = df[(df.season == s) & (df.ym == ym)]
            cur = df[(df.season == s) & (df.ym < ym)]
            wins = {"tum gecmis sezonlar": df[df.season < s], "yalniz onceki sezon": df[df.season == s - 1],
                    "gecmis + bu sezon": pd.concat([df[df.season < s], cur]),
                    "son 1230 mac (~1 sezon)": pd.concat([df[df.season < s], cur]).tail(1230),
                    "yalniz bu sezon": cur if len(cur) >= 300 else None}
            row = {"season": s, "ym": str(ym), "n": len(te)}
            for k, tr in wins.items():
                if tr is None:
                    row[k] = np.nan
                    continue
                r = bm_probs(te.mu, fit_bm(tr))
                row[k] = logloss(r, te.htft_reg, REG) * len(te)
            fr = df[df.season < s].htft_reg.value_counts(normalize=True).reindex(REG).fillna(1e-6)
            row["sabit frekans"] = logloss(pd.DataFrame([fr] * len(te), index=te.index), te.htft_reg, REG) * len(te)
            res.append(row)
    res = pd.DataFrame(res)
    cols = ["sabit frekans", "tum gecmis sezonlar", "yalniz onceki sezon", "gecmis + bu sezon", "son 1230 mac (~1 sezon)"]
    agg = res.groupby("season")[cols + ["n"]].sum()
    out = agg[cols].div(agg.n, axis=0)
    out.loc["toplam"] = res[cols].sum() / res.n.sum()
    print(out.round(4).to_string())
    ok = res.dropna(subset=["yalniz bu sezon"])
    print(f"\n'yalniz bu sezon' (en az 300 mac birikince; {ok.n.sum()} test maci): "
          f"{ok['yalniz bu sezon'].sum() / ok.n.sum():.4f}  vs ayni maclarda tum gecmis "
          f"{ok['tum gecmis sezonlar'].sum() / ok.n.sum():.4f}")
    d = (res["yalniz onceki sezon"] - res["tum gecmis sezonlar"]) / res.n
    print(f"Aylik fark (yalniz onceki sezon - tum gecmis): ortalama {d.mean():+.4f}, "
          f"t = {d.mean() / (d.std() / np.sqrt(len(d))):+.2f}  (negatif = yakin pencere daha iyi)")

    # ------------------------------------------------------------ S5
    h("S5. SEZON ICI KESIF / DOGRULAMA: ilk yarida bulunan etki ayni sezonun ikinci yarisinda tekrarliyor mu?")
    rows = []
    for s in seasons:
        g = df[df.season == s].drop(columns=[c for c in df if c.startswith("bm_")])
        first = g[g.half]
        full = add_features(g, fit_bm(first))          # model yalniz ilk yaridan: ikinci yari icin geleceği gormuyor
        for target, prob in (("r12", "bm_1/2"), ("r21", "bm_2/1")):
            full[target] = full.htft_reg == ("1/2" if target == "r12" else "2/1")
            for name, col in CANDS.items():
                rec = {"sezon": s, "hedef": prob[3:], "alan": name}
                for part, msk in (("ilk", full.half), ("ikinci", ~full.half)):
                    d = full[msk].dropna(subset=[col])
                    if d[col].nunique() < 2:
                        rec[f"{part}_z"], rec[f"{part}_p"] = np.nan, np.nan
                        continue
                    lr, coef = lr_test(d, target, prob, col)
                    rec[f"{part}_z"] = np.sign(coef) * np.sqrt(max(lr, 0))
                    rec[f"{part}_p"] = chi2.sf(max(lr, 0), 1)
                rows.append(rec)
    t = pd.DataFrame(rows).dropna()
    t["ilk_BH"] = False
    for s, g in t.groupby("sezon"):
        t.loc[g.index, "ilk_BH"] = bh(g.ilk_p)
    t["tekrar"] = t.ilk_BH & (np.sign(t.ilk_z) == np.sign(t.ikinci_z)) & (t.ikinci_p < 0.05)
    summ = t.groupby("sezon").apply(lambda g: pd.Series({
        "test": len(g), "ilk yari p<0.05": (g.ilk_p < 0.05).sum(), "ilk yari BH anlamli": g.ilk_BH.sum(),
        "ikinci yarida tekrarlayan": g.tekrar.sum(),
        "corr(ilk z, ikinci z)": np.corrcoef(g.ilk_z, g.ikinci_z)[0, 1]}), include_groups=False)
    print(summ.round(3).to_string())
    print(f"\nToplam {len(t)} test. Bilgi hic yoksa: p<0.05 icin ~{0.05 * len(t) / len(seasons):.1f}/sezon, "
          f"korelasyon ~0 beklenir.")
    top = t.sort_values("ilk_p").head(12)
    print("\nIlk yarilarda en guclu 12 etki ve ikinci yaridaki karsiliklari (z isareti = etkinin yonu):")
    print(top[["sezon", "hedef", "alan", "ilk_z", "ilk_p", "ilk_BH", "ikinci_z", "ikinci_p"]].round(4).to_string(index=False))

    # sezonlar arasi: bir sezonun etkisi sonraki sezonda devam ediyor mu? (tam sezon z'leri)
    zs = {}
    for s in seasons:
        g = df[df.season == s].drop(columns=[c for c in df if c.startswith("bm_")])
        full = add_features(g, fit_bm(df[df.season != s]))
        for target, prob in (("r12", "bm_1/2"), ("r21", "bm_2/1")):
            full[target] = full.htft_reg == prob[3:]
            for name, col in CANDS.items():
                d = full.dropna(subset=[col])
                if d[col].nunique() < 2:
                    continue
                lr, coef = lr_test(d, target, prob, col)
                zs[(prob[3:], name, s)] = np.sign(coef) * np.sqrt(max(lr, 0))
    z = pd.Series(zs).unstack()
    print("\nTam sezon z degerleri ve ardisik sezonlar arasi korelasyon (etki sezona ozgu ama kaliciysa > 0):")
    for a_, b_ in zip(seasons[:-1], seasons[1:]):
        print(f"  {a_} -> {b_}: corr = {z[a_].corr(z[b_]):+.3f}")
    big = z[(z.abs() > 2.5).any(axis=1)]
    if len(big):
        print("\n|z| > 2.5 olan alan-sezon ciftleri (diger sezonlardaki z ile):")
        print(big.round(2).to_string())
    print(f"\n{z.size} alan-sezon testi, |z|>2.5 sayisi {(z.abs() > 2.5).sum().sum()} "
          f"(bilgi yoksa beklenen ~{z.size * 2 * norm.sf(2.5):.1f})")


if __name__ == "__main__":
    main()
