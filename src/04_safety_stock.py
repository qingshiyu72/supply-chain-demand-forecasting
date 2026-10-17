"""04 预测 → 补货：把预测误差折算成安全库存，检验它到底能不能压缩库存。

问题：安全库存 SS = Z × σ × √L 里的 σ 该取哪个？
  传统做法：历史需求波动 σ_hist
  直觉做法：预测误差 σ_err —— "预测做得准，货就可以少备"

做法：三个 σ 口径各算一遍 SS，逐单元 + 分规模档对比。
  - σ全历史：训练集全程标准差
  - σ近98天：训练集最后 98 天标准差，窗口与预测跨度等长
  - σ误差：测试期逐日残差标准差

为什么必须同时看两个基准：全历史 σ 里混入了 2018→2021 的系统性水平下降，
直接拿它当基准，会把"趋势"误记成"预测带来的收益"。只有对齐到近 98 天的窗口，
才是同口径的比较。

为什么不做"库存缺口"：inventory_info 的 qty 与本数据的需求不是同一单位
（覆盖天数中位数 0.09 天，且与日均需求相关系数 0.96），不构成可信的运营水位，
因此仅作数据诊断记录，不参与结论。
"""

import pathlib
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
GROUPS = ["A", "B"]

WINDOW = 98                      # 与测试集跨度等长的对比窗口（天）
LEAD = 7                         # 补货提前期（天），数据未提供，按周补货假设
LEAD_SENS = [7, 14, 30]
Z_BY_ABC = {"A": 2.326, "B": 1.645, "C": 1.282}   # 99% / 95% / 90%
SERVICE_BY_ABC = {"A": 99, "B": 95, "C": 90}


def save(df, name):
    df.to_csv(OUT / name, index=False, encoding="utf-8-sig")


detail = pd.read_csv(OUT / "10_预测明细_测试集.csv")
unit_rows, summ_rows, sens_rows = [], [], []

for g in GROUPS:
    tr = pd.read_csv(DATA / g / f"use_train_{g}.csv", parse_dates=["ts"])
    te = detail[detail["组"] == g].copy()
    te["残差"] = te["qty"] - te["pred"]
    recent = tr[tr["ts"] >= tr["ts"].max() - pd.Timedelta(days=WINDOW - 1)]

    u = pd.concat([
        tr.groupby("unit")["qty"].std(ddof=1).rename("σ全历史"),
        recent.groupby("unit")["qty"].std(ddof=1).rename("σ近98天"),
        te.groupby("unit")["pred"].mean().rename("日均需求"),
        te.groupby("unit")["残差"].std(ddof=1).rename("σ误差"),
    ], axis=1)
    u.index.name = "unit"
    u = u.reset_index().dropna(subset=["σ全历史", "σ近98天", "σ误差"])

    # 规模档：与 02_forecast.py 一致，按单元日均需求三分位
    u["规模档"] = pd.qcut(u["日均需求"], 3, labels=["小", "中", "大"])

    # ABC 按预测需求累计占比分界（日均需求 × 98 天，常数不影响排序与占比）
    u = u.sort_values("日均需求", ascending=False).reset_index(drop=True)
    u["累计占比"] = u["日均需求"].cumsum() / u["日均需求"].sum()
    u["ABC"] = u["累计占比"].apply(lambda p: "A" if p <= 0.80 else ("B" if p <= 0.95 else "C"))
    u["服务水平%"] = u["ABC"].map(SERVICE_BY_ABC)
    u["Z值"] = u["ABC"].map(Z_BY_ABC)

    for col, name in [("σ全历史", "全历史"), ("σ近98天", "对齐"), ("σ误差", "误差")]:
        u[f"安全库存_{name}"] = (u["Z值"] * u[col] * np.sqrt(LEAD)).round(2)

    u["误差口径_相对对齐%"] = ((u["安全库存_误差"] / u["安全库存_对齐"] - 1) * 100).round(1)
    u["σ比_误差比近98天"] = (u["σ误差"] / u["σ近98天"]).round(3)

    s_all, s_ali, s_err = u["安全库存_全历史"].sum(), u["安全库存_对齐"].sum(), u["安全库存_误差"].sum()
    print(f"\n[{g}] 单元 {len(u)}｜ABC {u['ABC'].value_counts().to_dict()}")
    print(f"     安全库存合计：全历史口径 {s_all:,.0f}｜对齐口径 {s_ali:,.0f}｜误差口径 {s_err:,.0f}")
    print(f"     相对全历史口径 {(s_err / s_all - 1) * 100:+.1f}%｜"
          f"相对对齐口径 {(s_err / s_ali - 1) * 100:+.1f}%")
    print("     σ误差/σ近98天 中位（按规模档）：" +
          "、".join(f"{k} {d['σ比_误差比近98天'].median():.2f}"
                   for k, d in u.groupby("规模档", observed=True)))
    print("     预测误差反而大于近期波动的单元占比：" +
          f"{(u['σ比_误差比近98天'] > 1).mean() * 100:.0f}%")

    for k, d in u.groupby("规模档", observed=True):
        summ_rows.append({
            "组": g, "规模档": str(k), "单元数": len(d),
            "安全库存_全历史": round(d["安全库存_全历史"].sum(), 1),
            "安全库存_对齐": round(d["安全库存_对齐"].sum(), 1),
            "安全库存_误差": round(d["安全库存_误差"].sum(), 1),
            "误差相对对齐%": round((d["安全库存_误差"].sum() / d["安全库存_对齐"].sum() - 1) * 100, 1),
            "σ比中位": d["σ比_误差比近98天"].median().round(2),
        })

    # 敏感性：SS 正比于 √L，比值应与 L 无关 —— 用来证明结论不依赖提前期假设
    for L in LEAD_SENS:
        a = (u["Z值"] * u["σ近98天"] * np.sqrt(L)).sum()
        e = (u["Z值"] * u["σ误差"] * np.sqrt(L)).sum()
        sens_rows.append({"组": g, "提前期_天": L,
                          "安全库存_对齐": round(a, 1), "安全库存_误差": round(e, 1),
                          "误差相对对齐%": round((e / a - 1) * 100, 1)})

    u.insert(0, "组", g)
    unit_rows.append(u)

u = pd.concat(unit_rows, ignore_index=True)
keep = ["组", "unit", "规模档", "ABC", "服务水平%", "Z值", "日均需求", "累计占比",
        "σ全历史", "σ近98天", "σ误差", "安全库存_全历史", "安全库存_对齐", "安全库存_误差",
        "误差口径_相对对齐%", "σ比_误差比近98天"]
save(u[keep].round(3), "11_安全库存口径对比.csv")
save(pd.DataFrame(summ_rows), "12_安全库存汇总.csv")
save(pd.DataFrame(sens_rows), "13_提前期敏感性.csv")

print("\n" + "=" * 70)
print("按规模档汇总")
print("=" * 70)
print(pd.DataFrame(summ_rows).to_string(index=False))

print("\n" + "=" * 70)
print("提前期敏感性（安全库存 ∝ √L，比值应基本不变）")
print("=" * 70)
print(pd.DataFrame(sens_rows).to_string(index=False))

# ---- 数据诊断：为什么不用 inventory_info 做缺口分析
print("\n" + "=" * 70)
print("附：inventory_info 可用性诊断（未参与结论）")
print("=" * 70)
for g in GROUPS:
    inv = pd.read_csv(DATA / g / f"inventory_info_{g}.csv").set_index("unit")["qty"]
    avg = u[u["组"] == g].set_index("unit")["日均需求"]
    m = pd.concat([inv.rename("库存"), avg], axis=1).dropna()
    cov = (m["库存"] / m["日均需求"]).replace([np.inf, -np.inf], np.nan)
    print(f"[{g}] 覆盖天数中位 {cov.median():.3f} 天（远小于 1 天）｜库存~日均需求 相关系数 "
          f"{m['库存'].corr(m['日均需求']):.3f} → 量级上不可能是可运营的库存水位，无法支撑缺口分析")

print("\n已写出 11_安全库存口径对比.csv / 12_安全库存汇总.csv / 13_提前期敏感性.csv")
