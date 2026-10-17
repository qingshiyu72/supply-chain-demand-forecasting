"""01 数据画像与需求结构诊断（A/B 两条供应链）。

产出 outputs/00~03 四张表，回答四个问题：
  00 数据长什么样（规模、覆盖、稀疏程度）
  01 需求是否集中在少数单元（帕累托）
  02 需求水平逐年如何变化（用多长历史窗口才合理）
  03 需求是否存在周内规律（决定预测要不要做周内画像）
"""

import pathlib
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

GROUPS = ["A", "B"]
DOW = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def save(df, name):
    df.to_csv(OUT / name, index=False, encoding="utf-8-sig")


profile, conc, yearly, dow_tab = [], [], [], []

for g in GROUPS:
    tr = pd.read_csv(DATA / g / f"use_train_{g}.csv", parse_dates=["ts"])

    profile.append({
        "组": g,
        "订单行数": len(tr),
        "单元数": tr.unit.nunique(),
        "地理节点数": tr.geography.nunique(),
        "商品节点数": tr["product"].nunique(),
        "覆盖天数": tr.ts.nunique(),
        "起始日期": tr.ts.min().date(),
        "结束日期": tr.ts.max().date(),
        "日均需求": round(tr.qty.mean(), 1),
        "需求中位数": round(tr.qty.median(), 1),
        "零需求占比%": round((tr.qty == 0).mean() * 100, 2),
    })

    unit_sum = tr.groupby("unit").qty.sum().sort_values(ascending=False)
    share = pd.DataFrame({
        "组": g,
        "排名": range(1, len(unit_sum) + 1),
        "单元": unit_sum.index,
        "累计需求占比": (unit_sum.cumsum() / unit_sum.sum()).round(4).values,
    })
    conc.append(share)

    for year, v in tr.groupby(tr.ts.dt.year).qty.mean().items():
        yearly.append({"组": g, "年份": year, "日均需求": round(v, 1)})

    dow_mean = tr.groupby(tr.ts.dt.dayofweek).qty.mean()
    for i, v in dow_mean.items():
        dow_tab.append({"组": g, "星期": DOW[i], "日均需求": round(v, 1),
                        "相对指数": round(v / dow_mean.mean(), 3)})

    n80 = int((share["累计需求占比"] < 0.8).sum()) + 1
    n = tr.unit.nunique()
    print(f"[{g}] {len(tr):,} 行 / {n} 单元 / 日均 {tr.qty.mean():,.0f}")
    print(f"     贡献 80% 需求的单元：{n80} 个（占 {n80 / n:.1%}）")
    y = tr.groupby(tr.ts.dt.year).qty.mean()
    print("     各年日均：" + " → ".join(f"{yy}:{v:,.0f}" for yy, v in y.items()))
    print(f"     周内最大偏离：{(dow_mean.max() / dow_mean.min() - 1):.2%}")

save(pd.DataFrame(profile), "00_数据画像.csv")
save(pd.concat(conc, ignore_index=True), "01_需求集中度.csv")
save(pd.DataFrame(yearly), "02_年度水平.csv")
save(pd.DataFrame(dow_tab), "03_周内效应.csv")
print("\n已写出 00~03 四张表到 outputs/")
