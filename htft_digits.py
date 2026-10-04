"""1/2 ve 2/1 biten maclarda oranlarin basamaklari (birler, onlar, ondaliklar, toplamlar) ortak bir desen tasiyor mu?

Iki test:
 1) Her basamak ozelligi icin olabilirlik-orani testi: mac sonucu orani zaten modeldeyken ozellik ek bilgi veriyor mu?
    (Kontrol sart: orandaki basamaklar favorinin gucunun bir fonksiyonu, tek basina bakmak yaniltir.)
 2) Kesif / dogrulama: 2022-2023 sezonlarinda en guclu gorunen desenler 2024-2026'da da tutuyor mu?
Calistir: python htft_digits.py
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression

from backtest import DATA

TARGETS = {"1/2": "y12", "2/1": "y21"}


def digits(x, scale):
    """x*scale tam sayisinin (birler, onlar, basamak toplami) degerleri."""
    n = (x.abs() * scale).round().astype(int)
    return n % 10, (n // 10) % 10, n.astype(str).map(lambda s: sum(int(c) for c in s))


def load():
    df = pd.read_csv(DATA, dtype={"ft_result": str, "ht_result": str}, parse_dates=["date"])
    df = df.dropna(subset=["spread_home_points", "spread_home_decimal_odds", "spread_away_decimal_odds"]).reset_index(drop=True)
    df["y12"] = (df.htft == "1/2").astype(int)
    df["y21"] = (df.htft == "2/1").astype(int)
    f = {}
    dec = {"ms_ev": "money_home_decimal_odds", "ms_dep": "money_away_decimal_odds", "ust": "total_over_decimal_odds",
           "alt": "total_under_decimal_odds", "hnd_ev": "spread_home_decimal_odds", "hnd_dep": "spread_away_decimal_odds"}
    for name, col in dec.items():               # ondalik oran 1.62 -> yuzde birler=2, onda birler=6, tam kisim=1
        one, ten, s = digits(df[col], 100)
        f[f"{name}_oran_2.ondalik"], f[f"{name}_oran_1.ondalik"], f[f"{name}_oran_basamak_toplami"] = one, ten, s
        f[f"{name}_oran_tam_kisim"] = df[col].astype(int).clip(upper=5)
    for name, col in {"ms_ev": "money_home_odds", "ms_dep": "money_away_odds"}.items():   # Amerikan oran -160
        one, ten, s = digits(df[col], 1)
        f[f"{name}_amerikan_birler"], f[f"{name}_amerikan_onlar"], f[f"{name}_amerikan_basamak_toplami"] = one, ten, s
    one, ten, s = digits(df.total_over_points.astype(int).astype(float), 1)                # barem 226.5
    f["barem_birler"], f["barem_onlar"], f["barem_basamak_toplami"] = one, ten, s
    f["barem_bucuklu"] = (df.total_over_points % 1 != 0).astype(int)
    f["handikap_birler"] = df.spread_home_points.abs().astype(int) % 10
    f["handikap_bucuklu"] = (df.spread_home_points % 1 != 0).astype(int)
    # toplamlar
    ms_sum = df.money_home_decimal_odds + df.money_away_decimal_odds
    f["ms_oran_toplami_2.ondalik"], f["ms_oran_toplami_1.ondalik"], f["ms_oran_toplami_basamak_toplami"] = digits(ms_sum, 100)
    f["ms_oran_toplami_tam_kisim"] = ms_sum.astype(int).clip(upper=8)
    f["altust_oran_toplami_2.ondalik"] = digits(df.total_over_decimal_odds + df.total_under_decimal_odds, 100)[0]
    f["barem+handikap_birler"] = (df.total_over_points + df.spread_home_points.abs()).astype(int) % 10
    f["barem-handikap_birler"] = (df.total_over_points - df.spread_home_points.abs()).astype(int) % 10
    f["tum_oranlar_basamak_toplami_mod10"] = sum(f[f"{n}_oran_basamak_toplami"] for n in dec) % 10
    f["ms_ev+barem_basamak_toplami_mod10"] = (f["ms_ev_oran_basamak_toplami"] + f["barem_basamak_toplami"]) % 10
    f["ms_ev_2.ondalik==ms_dep_2.ondalik"] = (f["ms_ev_oran_2.ondalik"] == f["ms_dep_oran_2.ondalik"]).astype(int)
    f["ms_ev_ile_ms_dep_2.ondalik_toplami"] = (f["ms_ev_oran_2.ondalik"] + f["ms_dep_oran_2.ondalik"]) % 10
    feats = pd.DataFrame(f)
    return df, feats


def base_matrix(df):
    p = df.p_ml_home.to_numpy()
    return np.column_stack([p, (p - 0.5) ** 2, np.abs(p - 0.5)])


def loglik(X, y):
    m = LogisticRegression(C=1e6, max_iter=5000).fit(X, y)
    q = np.clip(m.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
    return (y * np.log(q) + (1 - y) * np.log(1 - q)).sum(), q


def lr_tests(df, feats):
    B = base_matrix(df)
    rows = []
    for tname, t in TARGETS.items():
        y = df[t].to_numpy()
        ll0, _ = loglik(B, y)
        for c in feats:
            v = feats[c]
            keep = v.map(v.value_counts()) >= 40          # nadir degerleri tek sinifta topla
            d = pd.get_dummies(v.where(keep, -1), drop_first=True).to_numpy(float)
            if d.shape[1] == 0:
                continue
            ll1, _ = loglik(np.column_stack([B, d]), y)
            stat = 2 * (ll1 - ll0)
            rows.append({"hedef": tname, "ozellik": c, "sinif": d.shape[1] + 1, "LR": stat, "p": chi2.sf(stat, d.shape[1]),
                         "ham_p": raw_p(v.where(keep, -1), y)})
    r = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    n = len(r)                                             # Benjamini-Hochberg duzeltmesi
    r["p_duzeltilmis"] = (r.p * n / (r.index + 1))[::-1].cummin()[::-1].clip(upper=1)
    return r


def raw_p(v, y):
    """Kontrolsuz ki-kare (orana gore duzeltme YOK) - yaniltici olabilecegini gostermek icin."""
    t = pd.crosstab(v, y)
    e = np.outer(t.sum(1), t.sum(0)) / t.to_numpy().sum()
    return chi2.sf(((t.to_numpy() - e) ** 2 / e).sum(), (t.shape[0] - 1) * (t.shape[1] - 1))


def discover_replicate(df, feats, top=12):
    """Kesif sezonlarinda beklenene gore en cok sapan (ozellik=deger) hucreleri, dogrulama sezonlarinda tekrar olc."""
    B = base_matrix(df)
    a, b = (df.season <= 2023).to_numpy(), (df.season >= 2024).to_numpy()
    out = []
    for tname, t in TARGETS.items():
        y = df[t].to_numpy()
        _, q = loglik(B, y)                                # mac sonucu oranina gore beklenen olasilik
        cells = []
        for c in feats:
            for val in feats[c].unique():
                m = (feats[c] == val).to_numpy()
                if (m & a).sum() < 60 or (m & b).sum() < 60:
                    continue
                o, e = y[m & a].sum(), q[m & a].sum()
                z = (o - e) / np.sqrt((q[m & a] * (1 - q[m & a])).sum())
                o2, e2 = y[m & b].sum(), q[m & b].sum()
                z2 = (o2 - e2) / np.sqrt((q[m & b] * (1 - q[m & b])).sum())
                cells.append({"hedef": tname, "desen": f"{c} = {val}", "kesif_mac": (m & a).sum(), "kesif_gercek/beklenen": o / e,
                              "kesif_z": z, "dogr_mac": (m & b).sum(), "dogr_gercek/beklenen": o2 / e2, "dogr_z": z2})
        c = pd.DataFrame(cells)
        print(f"\n{tname}: {len(c)} desen tarandi; kesif z ile dogrulama z korelasyonu = {c.kesif_z.corr(c.dogr_z):+.3f}")
        out.append(c.reindex(c.kesif_z.abs().sort_values(ascending=False).index).head(top))
    return pd.concat(out)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    df, feats = load()
    print(f"{len(df)} mac, {feats.shape[1]} basamak ozelligi\n")
    r = lr_tests(df, feats)
    print("== 1) Mac sonucu orani kontrol edilince ek bilgi (en guclu 15) ==")
    print(r.head(15).round(4).to_string(index=False))
    print(f"\nduzeltilmis p < 0.05 olan ozellik sayisi: {(r.p_duzeltilmis < 0.05).sum()} / {len(r)}")
    print(f"kontrolsuz (ham) p < 0.05 olan: {(r.ham_p < 0.05).sum()} / {len(r)}   <- oran kontrol edilmezse 'desen' sanilanlar")
    print("\n== 2) Kesif (2022-2023) -> dogrulama (2024-2026) ==")
    print(discover_replicate(df, feats).round(3).to_string(index=False))
