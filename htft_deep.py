"""1/2 ve 2/1 biten maclarin ortak yonleri: ayrintili analiz.

Bolumler
  A  Taban oranlar (uzatmalar dahil ve normal sureye gore)
  B  Favori cercevesi: 1/2-2/1'ler "favori coktu" mu "favori geri dondu" mu, favori gucune gore
  C  Brown hareketi modeli (Stern 1994): handikaptan 9 IY/MS seceneginin olasiligi, kalibrasyon,
     walk-forward log-loss karsilastirmasi (sabit frekans, kNN)
  D  Ek bilgi testleri: model olasiligi kontrol edilince her oran alani icin olabilirlik orani testi
     + Benjamini-Hochberg coklu test duzeltmesi
  E  Oyun akisi (mac sonrasi, tahmin icin kullanilamaz): donusler nasil oluyor, devre farkina gore donus olasiligi
  F  Adil oran tablosu (handikaba gore 9 secenek) + basabas ve Kelly
  G  Orneklem buyuklugu: bir avantaji gormek icin kac mac gerekir

Calistir: python htft_deep.py
Cikti   : ekrana rapor, data/fair_htft_by_spread.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2, multivariate_normal, norm
from sklearn.linear_model import LogisticRegression

from backtest import DATA, walk_forward
from backtest import load as load_bt

ROOT = DATA.parent
REG = ["1/1", "1/X", "1/2", "X/1", "X/X", "X/2", "2/1", "2/X", "2/2"]   # Iddaa sirasi (MS normal sure)
OT6 = ["1/1", "1/2", "2/1", "2/2", "X/1", "X/2"]                          # uzatmalar dahil
BIG = 60.0


def load():
    df = pd.read_csv(DATA, dtype={"ft_result": str, "ht_result": str}, parse_dates=["date"]).sort_values("date")
    df = df.dropna(subset=["spread_home_points"]).drop_duplicates("game_id").reset_index(drop=True)  # kaynakta 12 yinelenen mac
    df["H"] = df.ht_home - df.ht_away                                                  # devre farki (ev lehine)
    df["D"] = df[[f"q{i}_home" for i in range(1, 5)]].sum(axis=1) - df[[f"q{i}_away" for i in range(1, 5)]].sum(axis=1)
    df["S"] = df.D - df.H                                                              # ikinci yari farki
    df["F"] = df.score_home - df.score_away                                            # uzatmalar dahil
    df["mu"] = -df.spread_home_points                                                  # beklenen ev farki
    sgn = lambda x: np.select([x > 0, x < 0], ["1", "2"], "X")
    df["htft_reg"] = pd.Series(sgn(df.H)) + "/" + pd.Series(sgn(df.D))
    df["fav_home"] = df.mu > 0
    df["p_fav"] = np.where(df.fav_home, df.p_ml_home, 1 - df.p_ml_home)
    df["fav_spread"] = df.mu.abs()
    return df


# ---------------------------------------------------------------- C: Brown hareketi modeli
def fit_bm(tr):
    """Devre ve normal sure farki icin ortak normal dagilim parametreleri (en kucuk kareler)."""
    a, b = np.polyfit(tr.mu, tr.H, 1)                 # E[H] = a*mu + b
    c, d = np.polyfit(tr.mu, tr.D, 1)                 # E[D] = c*mu + d
    rh, rd = tr.H - (a * tr.mu + b), tr.D - (c * tr.mu + d)
    rs = rd - rh
    return {"a": a, "b": b, "c": c, "d": d, "sh": rh.std(), "sd": rd.std(), "ss": rs.std(),
            "rho": np.corrcoef(rh, rd)[0, 1], "rho_halves": np.corrcoef(rh, rs)[0, 1],
            "p_ot_home": (tr.F[tr.D == 0] > 0).mean()}


def bm_probs(mu, p):
    """Her mac icin normal sureye gore 9 IY/MS olasiligi (tamsayi farklar icin +-0.5 sureklilik duzeltmesi)."""
    mh, md = p["a"] * mu + p["b"], p["c"] * mu + p["d"]
    cov = [[1, p["rho"]], [p["rho"], 1]]
    cuts = {"1": (0.5, BIG), "X": (-0.5, 0.5), "2": (-BIG, -0.5)}
    F = lambda x, y: multivariate_normal.cdf(np.column_stack([x, y]), mean=[0, 0], cov=cov)
    out = {}
    for h, (h1, h2) in cuts.items():
        for f, (d1, d2) in cuts.items():
            zh1, zh2 = (h1 - mh) / p["sh"], (h2 - mh) / p["sh"]
            zd1, zd2 = (d1 - md) / p["sd"], (d2 - md) / p["sd"]
            out[f"{h}/{f}"] = np.clip(F(zh2, zd2) - F(zh1, zd2) - F(zh2, zd1) + F(zh1, zd1), 1e-9, 1)
    r = pd.DataFrame(out, index=mu.index)[REG]
    return r.div(r.sum(axis=1), axis=0)


def reg_to_ot(r, p):
    """Normal sure olasiliklarini uzatmalar dahil 6 secenege cevirir (beraberlik uzatmada bozulur)."""
    q = p["p_ot_home"]
    return pd.DataFrame({
        "1/1": r["1/1"] + q * r["1/X"], "1/2": r["1/2"] + (1 - q) * r["1/X"],
        "2/1": r["2/1"] + q * r["2/X"], "2/2": r["2/2"] + (1 - q) * r["2/X"],
        "X/1": r["X/1"] + q * r["X/X"], "X/2": r["X/2"] + (1 - q) * r["X/X"]}, index=r.index)


def logloss(P, labels, cats):
    idx = pd.Series(range(len(cats)), index=cats)[labels].to_numpy()
    return -np.log(np.clip(P.to_numpy()[np.arange(len(P)), idx], 1e-9, 1)).mean()


# ---------------------------------------------------------------- D: ek bilgi testleri
CANDS = {"alt/ust baremi (sezon ortalamasina gore)": "total_rel", "MS orani - handikap uyusmazligi": "ml_mismatch",
         "MS ve handikap marjsiz olasilik farki": "ml_vs_spread", "handikap oran farki (juice)": "spread_juice",
         "alt/ust oran farki (juice)": "total_juice", "MS marji (overround)": "overround_ml",
         "handikap^2 (dogrusalsizlik)": "mu2", "handikap x barem etkilesimi": "mu_x_total",
         "favori evde mi": "fav_home",
         "MS bahis yuzdesi (ev)": "money_home_stake_percentage", "MS para yuzdesi (ev)": "money_home_wager_percentage",
         "MS para-bahis farki (keskin para)": "money_sharp",
         "handikap bahis yuzdesi (ev)": "spread_home_stake_percentage", "handikap para yuzdesi (ev)": "spread_home_wager_percentage",
         "handikap keskin para": "spread_sharp", "ust bahis yuzdesi": "total_over_stake_percentage",
         "ust para yuzdesi": "total_over_wager_percentage", "alt/ust keskin para": "total_sharp"}


def add_features(df, p):
    """Brown modeli olasiliklari (bm_*) ve CANDS'teki aday oran alanlari."""
    full = df.join(bm_probs(df.mu, p).add_prefix("bm_"))
    full["total_rel"] = full.total_over_points - full.groupby("season").total_over_points.transform("mean")
    full["ml_vs_spread"] = np.log(full.p_ml_home / (1 - full.p_ml_home)) - np.log(full.p_spread_home / (1 - full.p_spread_home))
    pe = full.p_ml_home
    fit = np.polyfit(full.mu, np.log(pe / (1 - pe)), 3)          # handikaptan beklenen MS olasiligi
    full["ml_mismatch"] = np.log(pe / (1 - pe)) - np.polyval(fit, full.mu)
    full["spread_juice"] = full.spread_home_decimal_odds - full.spread_away_decimal_odds
    full["total_juice"] = full.total_over_decimal_odds - full.total_under_decimal_odds
    full["overround_ml"] = 1 / full.money_home_decimal_odds + 1 / full.money_away_decimal_odds - 1
    full["mu2"] = full.mu ** 2
    full["mu_x_total"] = full.mu * full.total_rel
    for k in ("money", "spread", "total"):
        side = "over" if k == "total" else "home"
        full[f"{k}_sharp"] = full[f"{k}_{side}_wager_percentage"] - full[f"{k}_{side}_stake_percentage"]
    return full


def lr_test(d, target, prob, col):
    """Olabilirlik orani istatistigi (Wilks: ~ ki-kare(1)) ve katsayi: model olasiligina col eklenince."""
    y = d[target].astype(int).to_numpy()
    base = np.log(d[prob] / (1 - d[prob])).to_numpy()[:, None]
    x = d[col].astype(float).to_numpy()
    x = ((x - x.mean()) / (x.std() or 1))[:, None]
    ll = []
    for X in (base, np.hstack([base, x])):
        m = LogisticRegression(C=1e6, max_iter=1000).fit(X, y)
        pr = np.clip(m.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
        ll.append((y * np.log(pr) + (1 - y) * np.log(1 - pr)).sum())
    return 2 * (ll[1] - ll[0]), m.coef_[0][1]


def bh(p, q=0.05):
    """Benjamini-Hochberg: FDR q duzeyinde anlamli olanlar (p ile ayni sirada boolean)."""
    p = np.asarray(p)
    order = np.argsort(p)
    ok = p[order] <= q * np.arange(1, len(p) + 1) / len(p)
    keep = np.zeros(len(p), bool)
    if ok.any():
        keep[order[: np.nonzero(ok)[0].max() + 1]] = True
    return keep


# ---------------------------------------------------------------- yazdirma yardimcilari
def h(t):
    print(f"\n{'=' * 100}\n{t}\n{'=' * 100}")


def pct(x):
    return f"{100 * x:5.2f}%"


def main():
    pd.set_option("display.width", 220)
    df = load()
    n = len(df)
    df["y12"], df["y21"] = df.htft == "1/2", df.htft == "2/1"
    df["r12"], df["r21"] = df.htft_reg == "1/2", df.htft_reg == "2/1"

    # ------------------------------------------------------------ A
    h(f"A. TABAN ORANLAR  ({n} mac, {df.date.min():%Y-%m} - {df.date.max():%Y-%m}, BetMGM kapanis)")
    a = pd.DataFrame({"uzatmalar dahil": df.htft.value_counts(normalize=True),
                      "normal sure (Iddaa)": df.htft_reg.value_counts(normalize=True)}).reindex(REG).fillna(0)
    a["adil oran (normal sure)"] = 1 / a["normal sure (Iddaa)"]
    print(a.round(4).to_string())
    print(f"\nUzatmaya giden mac: {pct((df.D == 0).mean())}.  Sezona gore 1/2 ve 2/1 (uzatmalar dahil):")
    print(df.groupby("season")[["y12", "y21"]].mean().round(4).assign(mac=df.groupby("season").size()).to_string())

    # ------------------------------------------------------------ B
    h("B. FAVORI CERCEVESI: donusler kimin lehine?")
    fav_led = np.where(df.fav_home, df.H > 0, df.H < 0)
    fav_won = np.where(df.fav_home, df.F > 0, df.F < 0)
    rev = df.y12 | df.y21
    df["fav_collapse"] = rev & fav_led & ~fav_won      # favori devrede onde, maci kaybetti
    df["fav_comeback"] = rev & ~fav_led & fav_won      # favori devrede geride, maci kazandi
    print(f"Donus (1/2 veya 2/1): {rev.sum()} mac = {pct(rev.mean())}")
    print(f"  favori geri dondu (devrede geride, kazandi): {df.fav_comeback.sum():5d}  {pct(df.fav_comeback.mean())}")
    print(f"  favori coktu      (devrede onde, kaybetti) : {df.fav_collapse.sum():5d}  {pct(df.fav_collapse.mean())}")
    bins = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0]
    g = df.groupby([pd.cut(df.p_fav, bins), "fav_home"], observed=True)
    t = g.agg(mac=("htft", "size"), fav_handikap=("fav_spread", "mean"), donus=("fav_collapse", lambda s: 0),
              favori_geri_dondu=("fav_comeback", "mean"), favori_coktu=("fav_collapse", "mean"),
              y12=("y12", "mean"), y21=("y21", "mean"))
    t["donus"] = t.favori_geri_dondu + t.favori_coktu
    print("\nFavorinin kazanma olasiligina (MS oranindan) ve favorinin evde/deplasmanda olmasina gore:")
    print(t.round(3).to_string())
    print("\nOkuma: favori guclendikce 'favori coktu' azalir, 'favori geri dondu' bir noktaya kadar artar."
          "\n1/2 = ev sahibi devrede onde & deplasman kazanir; favori deplasmandaysa bu 'favori geri dondu' demektir.")

    # ------------------------------------------------------------ C
    h("C. BROWN HAREKETI MODELI (Stern 1994) - handikaptan 9 IY/MS olasiligi")
    p = fit_bm(df)
    print(f"E[devre farki]       = {p['a']:.3f} * beklenen_fark + {p['b']:+.2f}   (teori: 0.5 * beklenen_fark)")
    print(f"E[normal sure farki] = {p['c']:.3f} * beklenen_fark + {p['d']:+.2f}   (teori: 1.0 * beklenen_fark)")
    print(f"sapma: devre sigma={p['sh']:.2f}, ikinci yari sigma={p['ss']:.2f}, normal sure sigma={p['sd']:.2f}"
          f"   (teori: devre = mac/sqrt(2) = {p['sd'] / np.sqrt(2):.2f})")
    print(f"corr(devre, mac sonu) = {p['rho']:.3f}  (teori sqrt(1/2) = 0.707);  "
          f"corr(1. yari, 2. yari) = {p['rho_halves']:+.3f}  (teori 0: bagimsiz artislar)")
    print(f"uzatmada ev sahibi kazanma orani = {pct(p['p_ot_home'])}")
    # sigma toplamla olceklenir mi? (puan sayisi arttikca fark varyansi artar: sigma ~ sqrt(toplam))
    rd = df.D - (p["c"] * df.mu + p["d"])
    tq = pd.qcut(df.total_over_points, 5)
    s_by_total = df.assign(rd=rd).groupby(tq, observed=True).rd.std()
    print("\nNormal sure fark sapmasi, alt/ust baremine gore (Brown: sigma ~ sqrt(toplam)):")
    print(pd.DataFrame({"sigma": s_by_total, "sqrt-olcekli beklenti": p["sd"] * np.sqrt(
        df.groupby(tq, observed=True).total_over_points.mean() / df.total_over_points.mean())}).round(2).to_string())

    # walk-forward: her test sezonu yalnizca onceki sezonlarla
    parts = []
    for s in sorted(x for x in df.season.unique() if x >= 2024):
        tr, te = df[df.season < s], df[df.season == s]
        pp = fit_bm(tr)
        r = bm_probs(te.mu, pp)
        parts.append(te[["season", "htft", "htft_reg", "mu"]].join(r.add_prefix("bm_")).join(
            reg_to_ot(r, pp).add_prefix("bmot_")))
    wf = pd.concat(parts)
    base_reg = df[df.season < 2024].htft_reg.value_counts(normalize=True).reindex(REG).fillna(1e-9)
    base_ot = df[df.season < 2024].htft.value_counts(normalize=True).reindex(OT6).fillna(1e-9)
    knn = walk_forward(load_bt())
    knn = knn[knn.game_id.isin(df.game_id[wf.index])]
    print(f"\nWalk-forward log-loss (test 2024-2026, {len(wf)} mac; dusuk = iyi):")
    print(f"  normal sure 9 secenek : sabit frekans {logloss(pd.DataFrame([base_reg] * len(wf)), wf.htft_reg, REG):.4f}"
          f"   Brown modeli {logloss(wf[[f'bm_{c}' for c in REG]], wf.htft_reg, REG):.4f}")
    print(f"  uzatmalar dahil 6     : sabit frekans {logloss(pd.DataFrame([base_ot] * len(wf)), wf.htft, OT6):.4f}"
          f"   Brown modeli {logloss(wf[[f'bmot_{c}' for c in OT6]], wf.htft, OT6):.4f}"
          f"   kNN (k=500) {logloss(knn[[f'pred_{c}' for c in OT6]], knn.htft, OT6):.4f}")
    print("\nKalibrasyon (walk-forward, Brown modeli tahminine gore 5 dilim):")
    for c in ("1/2", "2/1"):
        q = pd.qcut(wf[f"bm_{c}"], 5)
        k = wf.groupby(q, observed=True).agg(mac=("htft_reg", "size"), tahmin=(f"bm_{c}", "mean"),
                                              gercek=("htft_reg", lambda s: (s == c).mean()))
        k["z"] = (k.gercek - k.tahmin) / np.sqrt(k.tahmin * (1 - k.tahmin) / k.mac)
        print(f"\n{c} (normal sure):\n{k.round(4).to_string()}")

    # ------------------------------------------------------------ D
    h("D. EK BILGI TESTLERI: Brown modeli (handikap) kontrol edilince baska oran alani bilgi tasiyor mu?")
    full = add_features(df, p)
    rows = []
    for target, prob in (("r12", "bm_1/2"), ("r21", "bm_2/1")):
        for name, col in CANDS.items():
            d = full.dropna(subset=[col])
            lr, coef = lr_test(d, target, prob, col)
            rows.append({"hedef": "1/2" if target == "r12" else "2/1", "alan": name, "mac": len(d),
                         "LR": lr, "p": chi2.sf(lr, 1), "katsayi": coef})
    t = pd.DataFrame(rows).sort_values("p")
    m_ = len(t)
    t["BH_esik"] = 0.05 * np.arange(1, m_ + 1) / m_
    t["BH_anlamli"] = t.p <= t.BH_esik
    t["BH_anlamli"] = t.BH_anlamli[::-1].cummax()[::-1]
    print(t.round(4).to_string(index=False))
    print(f"\n{m_} test; Benjamini-Hochberg (FDR %5) sonrasi anlamli: {t.BH_anlamli.sum()}")
    print(f"Duzeltmesiz p<0.05: {(t.p < 0.05).sum()} (hic bilgi olmasa bile beklenen ~{0.05 * m_:.1f})")

    # ------------------------------------------------------------ E
    h("E. OYUN AKISI (mac sonrasi istatistikler - mac oncesi tahmin icin kullanilamaz)")
    df["q3"] = df.q3_home - df.q3_away
    df["q4"] = df.q4_home - df.q4_away
    df["fav_H"] = np.where(df.fav_home, df.H, -df.H)
    for lab, msk in (("1/2", df.y12), ("2/1", df.y21), ("diger", ~(df.y12 | df.y21))):
        x = df[msk]
        lead = x.H.abs()
        print(f"{lab:6s} mac={len(x):5d}  devre farki ort={lead.mean():5.2f} medyan={lead.median():4.1f}  "
              f"uzatmaya giden={pct((x.D == 0).mean())}  mutlak ikinci yari farki ort={x.S.abs().mean():5.2f}  "
              f"donus 4. ceyrekte tamamlandi={pct(((x.H + x.q3) * x.H > 0).mean()) if lab != 'diger' else '   - '}")
    print("\nDevre farkina gore donus olasiligi (gercek vs Brown: P = Phi(-(|H| + beklenen 2. yari) / sigma_2yari)):")
    df["lead"] = df.H.abs().clip(upper=20)
    df["leader_won"] = np.sign(df.F) == np.sign(df.H)
    exp2 = (p["c"] - p["a"]) * df.mu + (p["d"] - p["b"])          # ikinci yarida beklenen ev farki
    df["bm_rev"] = norm.cdf(-(df.H.abs() + np.sign(df.H) * exp2) / p["ss"])
    e = df[df.H != 0].groupby("lead").agg(mac=("H", "size"), gercek_donus=("leader_won", lambda s: 1 - s.mean()),
                                         brown=("bm_rev", "mean"))
    print(e.round(3).to_string())

    # ------------------------------------------------------------ F
    h("F. ADIL ORAN TABLOSU (normal sure, Iddaa'nin 9 secenegi) + BASABAS")
    grid = pd.Series(np.arange(-14.5, 15.0, 1.0), name="ev_handikap")
    r = bm_probs(pd.Series(-grid.to_numpy()), p)
    fair = (1 / r).round(2)
    fair.insert(0, "ev_handikap", grid.to_numpy())
    fair.to_csv(ROOT / "fair_htft_by_spread.csv", index=False)
    print(fair[fair.ev_handikap.isin([-12.5, -9.5, -7.5, -5.5, -3.5, -1.5, 1.5, 3.5, 5.5, 7.5, 9.5, 12.5])].to_string(index=False))
    print(f"\nyazildi: {ROOT / 'fair_htft_by_spread.csv'}")
    print("\nBasabas: bir oran ancak (site orani) > (adil oran) ise uzun vadede kazandirir.")
    print("Kelly orani f = (p*o - 1) / (o - 1); p = model olasiligi, o = site orani. Ornek, adil 6.50 (p=0.154):")
    for o in (5.7, 6.5, 7.0, 7.5, 8.0):
        pr = 1 / 6.5
        print(f"  site orani {o:4.2f}: beklenen getiri {100 * (pr * o - 1):+6.1f}%   Kelly {max(0, (pr * o - 1) / (o - 1)) * 100:4.1f}% kasa")

    # ------------------------------------------------------------ G
    h("G. ORNEKLEM BUYUKLUGU (merkezi limit teoremi): bir getiri avantajini %95 guvenle gormek icin")
    for pr, o in ((0.13, 7.5), (0.13, 8.5)):
        sd = o * np.sqrt(pr * (1 - pr))
        for edge in (0.05, 0.10, 0.20):
            print(f"  p={pr:.2f}, oran {o}: %{edge * 100:.0f} avantaj icin ~{int((1.96 * sd / edge) ** 2):6d} bahis")


if __name__ == "__main__":
    main()
