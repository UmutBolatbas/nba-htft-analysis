"""Gercek IDDAA oranlariyla (Mackolik arsivi) 1/2 ve 2/1 analizi.

  I1  Iddaa IY/MS marji ve oranlarin ima ettigi olasiliklar, sezon sezon
  I2  Kalibrasyon: Iddaa'nin 1/2 ve 2/1 orani (marjsiz) gerceklesen sikligi dogru mu veriyor? Oran araliklarina gore
  I3  Duz stratejiler: her 1/2, her 2/1, oran araliklarina gore -- sezon sezon getiri
  I4  Deger stratejisi: Iddaa orani > Brown modeli adil orani (BetMGM handikabindan, walk-forward) ise oyna
  I5  Iddaa oran yapisi (1/1, 2/2, 1/2, 2/1, MS1, MS2, handikap, barem, ilk yari) Brown modelinin otesinde
      bilgi tasiyor mu? Olabilirlik orani testleri + Benjamini-Hochberg, sezon ici kesif/dogrulama

IY/MS'de MS normal sureye gore (uzatma yok; X secenekleri var); sonuc ceyrek skorlarindan hesaplanir.
Calistir: python htft_iddaa.py
Girdi   : data/raw/mackolik/{matches,odds}.csv, data/nba_betmgm_closing.csv
Cikti   : data/nba_iddaa.csv
"""
import re

import numpy as np
import pandas as pd
from scipy.stats import chi2

from htft_deep import REG, ROOT, bh, bm_probs, fit_bm, h, load, lr_test, pct

MK = ROOT / "raw" / "mackolik"
OUT = ROOT / "nba_iddaa.csv"
NICK = {  # Mackolik takim adi -> kisaltma (teams.py ile ayni)
    "Atlanta Hawks": "atl", "Boston Celtics": "bos", "Brooklyn Nets": "bkn", "Charlotte Hornets": "cha",
    "Chicago Bulls": "chi", "Cleveland Cavaliers": "cle", "Dallas Mavericks": "dal", "Denver Nuggets": "den",
    "Detroit Pistons": "det", "Golden State Warriors": "gs", "Houston Rockets": "hou", "Indiana Pacers": "ind",
    "Los Angeles Clippers": "lac", "Los Angeles Lakers": "lal", "Memphis Grizzlies": "mem", "Miami Heat": "mia",
    "Milwaukee Bucks": "mil", "Minnesota Timberwolves": "min", "New Orleans Pelicans": "no", "New York Knicks": "ny",
    "Oklahoma City Thunder": "okc", "Orlando Magic": "orl", "Philadelphia 76ers": "phi", "Phoenix Suns": "phx",
    "Portland Trail Blazers": "por", "Sacramento Kings": "sac", "San Antonio Spurs": "sa", "Toronto Raptors": "tor",
    "Utah Jazz": "utah", "Washington Wizards": "wsh"}


def line_of(market):
    m = re.search(r"\(([\d,]+):([\d,]+)\)", market)
    if m:  # handikap (ev:deplasman baslangic sayisi) -> ev sahibi cizgisi
        a, b = (float(x.replace(",", ".")) for x in m.groups())
        return a - b
    m = re.search(r"\(([\d,]+)\)", market)
    return float(m.group(1).replace(",", ".")) if m else np.nan


def main_line(o, prefix):
    """Oranlari birbirine en yakin cizgi (ana cizgi)."""
    if o.empty:
        return pd.DataFrame()
    w = o.pivot_table(index=["match_id", "line"], columns="label", values="odds", aggfunc="last").reset_index()
    a, b = w.columns[-2], w.columns[-1]
    w["gap"] = (w[a] - w[b]).abs()
    w = w.sort_values("gap").groupby("match_id").head(1).set_index("match_id")
    return w[["line", a, b]].rename(columns={"line": f"{prefix}_line", a: f"{prefix}_{a}", b: f"{prefix}_{b}"})


def build():
    m = pd.read_csv(MK / "matches.csv", dtype={"match_id": str})
    o = pd.read_csv(MK / "odds.csv", dtype={"match_id": str, "label": str})
    o["odds"] = pd.to_numeric(o["odds"], errors="coerce")
    o = o.dropna(subset=["odds"])
    o["line"] = o["market"].map(line_of)
    base = o["market"].str.replace(r"\s*\(.*\)", "", regex=True).str.strip()

    iyms = o[base == "İlk Yarı/Maç Sonucu"].pivot_table(index="match_id", columns="label", values="odds", aggfunc="last")
    iyms = iyms.reindex(columns=REG).add_prefix("iy_")
    ms = o[(base == "Maç Sonucu") & o["label"].isin(["1", "2"])].pivot_table(
        index="match_id", columns="label", values="odds", aggfunc="last").add_prefix("ms")
    fh = o[base == "1. Yarı Sonucu"].pivot_table(index="match_id", columns="label", values="odds", aggfunc="last")
    fh = fh.reindex(columns=["1", "X", "2"]).add_prefix("iy1x2_")
    hnd = main_line(o[base == "Handikaplı Maç Sonucu"], "hnd")              # hnd_line = ev cizgisi, hnd_1 / hnd_2
    tot = main_line(o[base == "Maç Sonucu Alt/Üst"], "au")                  # au_line, au_Alt, au_Üst
    mbs = o[base == "İlk Yarı/Maç Sonucu"].groupby("match_id")["mbs"].first().rename("iy_mbs")
    odds = iyms.join([ms, fh, hnd, tot, mbs], how="left")

    m = m.join(odds, on="match_id", how="inner")
    m["home"], m["away"] = m["home"].map(NICK), m["away"].map(NICK)
    m["date_tr"] = pd.to_datetime(m["date"], format="%d.%m.%Y")
    m["season"] = m["season"].str[-4:].astype(int)                            # "2021/2022" -> 2022 (BetMGM ile ayni)
    sgn = lambda x: pd.Series(np.select([x > 0, x < 0], ["1", "2"], "X"), index=x.index)
    hts = m["ht"].str.extract(r"(\d+)\s*-\s*(\d+)").astype(float)
    fts = m["ft"].str.extract(r"(\d+)\s*-\s*(\d+)").astype(float)
    ms_reg = sgn(fts[0] - fts[1]).where(m["status"] != "UZ", "X")             # uzatmali mac normal surede berabere
    m["htft_mk"] = sgn(hts[0] - hts[1]) + "/" + ms_reg

    bt = load().drop(columns="season")                                        # BetMGM + ceyrekler + H, D, mu ...
    parts, left = [], m
    for shift in (1, 0, 2):                                                    # Mackolik tarihi TR saatine gore
        x = left.assign(date=left["date_tr"] - pd.Timedelta(days=shift))
        j = x.merge(bt, on=["date", "home", "away"], how="left", indicator=True, suffixes=("", "_bt"))
        parts.append(j[j["_merge"] == "both"].drop(columns="_merge"))
        left = j[j["_merge"] == "left_only"][m.columns]
    n_bt = sum(len(p) for p in parts)
    left = left.assign(date=left["date_tr"] - pd.Timedelta(days=1))
    df = pd.concat(parts + [left]).drop_duplicates("match_id").sort_values("date").reset_index(drop=True)
    df["betmgm"] = df["spread_home_points"].notna()
    both = df[df.betmgm]
    print(f"Mackolik oranli mac: {len(m)}, BetMGM/ceyrek verisiyle eslesen: {n_bt}, yalniz Mackolik: {len(left)}")
    print(f"Sonuc kontrolu (eslesenlerde Mackolik IY/MS = ceyreklerden IY/MS): {(both.htft_mk == both.htft_reg).mean():.4f}")
    df["htft_reg"] = df["htft_reg"].fillna(df["htft_mk"])
    df["mu"] = df["mu"].fillna(-df["hnd_line"])                                # BetMGM yoksa Iddaa handikabi (ev cizgisi)
    df.to_csv(OUT, index=False)
    return df


def implied(df):
    cols = [f"iy_{c}" for c in REG]
    inv = 1 / df[cols]
    df["iy_overround"] = inv.sum(axis=1) - 1
    for c in REG:
        df[f"q_{c}"] = inv[f"iy_{c}"] / inv.sum(axis=1)                       # marjsiz Iddaa olasiligi
    return df


def roi(mask, df, c):
    x = df[mask & df[f"iy_{c}"].notna()]
    win = x["htft_reg"] == c
    pl = np.where(win, x[f"iy_{c}"] - 1, -1.0)
    return len(x), win.mean() if len(x) else np.nan, pl.mean() if len(x) else np.nan


def main():
    pd.set_option("display.width", 220)
    df = implied(build())
    df = df[df["iy_1/2"].notna() & df["iy_2/1"].notna()].copy()
    seasons = sorted(df.season.unique())

    # ------------------------------------------------------------ I1
    h(f"I1. IDDAA IY/MS MARJI VE OLASILIKLAR ({len(df)} mac)")
    t = df.groupby("season").agg(mac=("htft_reg", "size"), iyms_marji=("iy_overround", "mean"),
                                 ort_1_2_oran=("iy_1/2", "mean"), ort_2_1_oran=("iy_2/1", "mean"),
                                 gercek_1_2=("htft_reg", lambda s: (s == "1/2").mean()),
                                 gercek_2_1=("htft_reg", lambda s: (s == "2/1").mean()),
                                 mbs1_orani=("iy_mbs", lambda s: (s == 1).mean()))
    print(t.round(3).to_string())
    print("\nSecenek bazinda: Iddaa'nin ima ettigi (marjsiz) olasilik vs gercek siklik, ortalama oran:")
    print(pd.DataFrame({"iddaa_olasilik": [df[f"q_{c}"].mean() for c in REG],
                        "gercek": [(df.htft_reg == c).mean() for c in REG],
                        "ort_oran": [df[f"iy_{c}"].mean() for c in REG],
                        "her_mac_oynasan_getiri": [roi(df.index == df.index, df, c)[2] for c in REG]},
                       index=REG).round(4).to_string())

    # ------------------------------------------------------------ I2
    h("I2. KALIBRASYON: Iddaa'nin 1/2 ve 2/1 oranina gore gercek siklik (ham oran dilimleri)")
    for c in ("1/2", "2/1"):
        bins = [0, 4.5, 5.5, 6.5, 7.5, 9, 11, 15, 25, 1000]
        g = df.groupby(pd.cut(df[f"iy_{c}"], bins), observed=True)
        k = g.agg(mac=("htft_reg", "size"), ort_oran=(f"iy_{c}", "mean"), iddaa_marjsiz=(f"q_{c}", "mean"),
                  gercek=("htft_reg", lambda s: (s == c).mean()))
        k["basabas"] = 1 / k.ort_oran
        k["getiri"] = [roi(df[f"iy_{c}"].between(i.left, i.right, inclusive="right"), df, c)[2] for i in k.index]
        k["z"] = (k.gercek - k.basabas) / np.sqrt(k.basabas * (1 - k.basabas) / k.mac)
        print(f"\n{c}:\n{k.round(4).to_string()}")
    print("\nz: gercek siklik basabas olasiligindan (1/oran) ne kadar uzak; z > 2 kazandiran dilim olurdu.")

    # ------------------------------------------------------------ I3
    h("I3. DUZ STRATEJILER SEZON SEZON (1 birim, getiri = ortalama kar/bahis)")
    rows = []
    for s in seasons + ["hepsi"]:
        msk = df.season == s if s != "hepsi" else df.index == df.index
        r = {"sezon": s}
        for c in ("1/2", "2/1"):
            n, w, g = roi(msk, df, c)
            r[f"{c} mac"], r[f"{c} isabet"], r[f"{c} getiri"] = n, w, g
            fav_home = df.mu > 0
            n2, w2, g2 = roi(msk & (fav_home if c == "2/1" else ~fav_home), df, c)  # favorinin geri donusu
            r[f"{c} favori-donus getiri"] = g2
        rows.append(r)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("'favori-donus' = 2/1 yalniz ev sahibi favoriyken, 1/2 yalniz deplasman favoriyken.")

    # ------------------------------------------------------------ I4
    h("I4. DEGER STRATEJISI: Iddaa orani > Brown adil orani x (1 + esik)  (walk-forward)")
    full = load()
    parts = []
    for s in seasons:
        te = df[df.season == s]
        tr = full[full.season < s]
        if len(tr) < 1000:
            continue
        parts.append(te.join(bm_probs(te.mu, fit_bm(tr)).add_prefix("bm_")))
    if not parts:
        print("Onceki sezonu olan test sezonu yok (en az 1000 egitim maci gerekir).")
        return
    wf = pd.concat(parts)
    rows = []
    for c in ("1/2", "2/1"):
        for e in (0.0, 0.05, 0.10, 0.20, 0.30):
            ev = wf[f"bm_{c}"] * wf[f"iy_{c}"] - 1
            msk = ev > e
            n, w, g = roi(msk, wf, c)
            rows.append({"secenek": c, "esik": e, "bahis": n, "isabet": w, "model_beklenen": ev[msk].mean() if n else np.nan,
                         "gercek_getiri": g, "sezonlar": ", ".join(f"{s}:{roi(msk & (wf.season == s), wf, c)[2]:+.2f}"
                                                                    for s in sorted(wf.season.unique()) if (msk & (wf.season == s)).any())})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    gap = (1 / wf[["bm_1/2", "bm_2/1"]].rename(columns=lambda x: x[3:]) - wf[["iy_1/2", "iy_2/1"]].values)
    print(f"\nTest maclari {len(wf)} (sezonlar {sorted(wf.season.unique())}). Ortalama (adil - Iddaa) oran farki: "
          f"1/2 {gap['1/2'].mean():+.2f}, 2/1 {gap['2/1'].mean():+.2f}")

    # ------------------------------------------------------------ I5
    h("I5. IDDAA ORAN YAPISI BROWN MODELININ (BetMGM handikabi) OTESINDE BILGI TASIYOR MU?")
    full_bm = df.join(bm_probs(df.mu, fit_bm(full)).add_prefix("bm_"))
    lg = lambda p: np.log(p / (1 - p))
    f = full_bm
    f["iddaa_q12_vs_model"] = lg(f["q_1/2"]) - lg(f["bm_1/2"])
    f["iddaa_q21_vs_model"] = lg(f["q_2/1"]) - lg(f["bm_2/1"])
    f["iddaa_q11_vs_model"] = lg(f["q_1/1"]) - lg(f["bm_1/1"])
    f["iddaa_q22_vs_model"] = lg(f["q_2/2"]) - lg(f["bm_2/2"])
    f["ms1_ms2_log_orani"] = np.log(f["ms1"] / f["ms2"])
    f["iddaa_vs_betmgm_handikap"] = f["hnd_line"] - f["spread_home_points"]   # ikisi de ev cizgisi (ayni isaret); fark
    f["iddaa_vs_betmgm_barem"] = f["au_line"] - f["total_over_points"]
    f["iy_overround"] = f["iy_overround"]
    f["iy1x2_beraberlik"] = 1 / f["iy1x2_X"]
    f["oran_12_eksi_21"] = f["iy_1/2"] - f["iy_2/1"]
    f["oran_11_x_22"] = np.log(f["iy_1/1"] * f["iy_2/2"])
    f["oran_12_x_21"] = np.log(f["iy_1/2"] * f["iy_2/1"])
    cands = ["iddaa_q12_vs_model", "iddaa_q21_vs_model", "iddaa_q11_vs_model", "iddaa_q22_vs_model",
             "ms1_ms2_log_orani", "iddaa_vs_betmgm_handikap", "iddaa_vs_betmgm_barem", "iy_overround",
             "iy1x2_beraberlik", "oran_12_eksi_21", "oran_11_x_22", "oran_12_x_21", "iy_1/2", "iy_2/1", "iy_1/1", "iy_2/2"]
    f["r12"], f["r21"] = f.htft_reg == "1/2", f.htft_reg == "2/1"
    rows = []
    for target, prob in (("r12", "bm_1/2"), ("r21", "bm_2/1")):
        for col in cands:
            d = f.dropna(subset=[col])
            if d[col].nunique() < 2:
                continue
            lr, coef = lr_test(d, target, prob, col)
            rec = {"hedef": prob[3:], "alan": col, "mac": len(d), "LR": lr, "p": chi2.sf(max(lr, 0), 1), "katsayi": coef}
            for s in seasons:                                                  # sezon sezon yon (z)
                ds = d[d.season == s]
                if len(ds) > 300:
                    l2, c2 = lr_test(ds, target, prob, col)
                    rec[f"z{s}"] = np.sign(c2) * np.sqrt(max(l2, 0))
            rows.append(rec)
    t = pd.DataFrame(rows).sort_values("p")
    t["BH_anlamli"] = bh(t.p)
    print(t.round(3).to_string(index=False))
    print(f"\n{len(t)} test; Benjamini-Hochberg sonrasi anlamli: {t.BH_anlamli.sum()}  "
          f"(duzeltmesiz p<0.05: {(t.p < 0.05).sum()}, sansla beklenen ~{0.05 * len(t):.1f})")
    print("z<sezon> sutunlari: etkinin her sezondaki yonu ve gucu; gercek bir yapi her sezon ayni isarette olmali.")


if __name__ == "__main__":
    main()
