"""02 需求预测：四种基线方法对比 + 误差分解。

流程（每个供应链组独立跑）：
  1) 验证集 = 训练集最后 98 天（2020-11-24 ~ 2021-03-01），与测试集预测跨度等长
  2) 四种方法在验证集上打分，选出加权 WAPE 最低的一个
  3) 用全量训练集重新拟合，预测测试集（2021-03-02 ~ 2021-06-07）
  4) 把最优方法的误差按 预测周 / 商品 / 单元规模 三个维度拆开

指标说明：weight_X.csv 给每个单元一个权重，因此主指标是加权 WAPE（业务量大的单元错得更贵）。
"""

import pathlib
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
GROUPS = ["A", "B"]
METHODS = ["全历史均值", "近90天均值", "近28天均值", "指数平滑(α=0.1)"]
VAL_DAYS = 98
ALPHA = 0.1


def save(df, name):
    df.to_csv(OUT / name, index=False, encoding="utf-8-sig")


def ses(qty):
    level = float(qty.iloc[0])
    for v in qty.iloc[1:]:
        level = ALPHA * float(v) + (1 - ALPHA) * level
    return level


def unit_forecast(fit, method):
    """每个单元给出一个常数预测值（数据无周内效应，见 03_周内效应.csv）。"""
    fit = fit.sort_values("ts")
    if method == "全历史均值":
        return fit.groupby("unit").qty.mean().rename("pred")
    if method == "近90天均值":
        return fit.groupby("unit").tail(90).groupby("unit").qty.mean().rename("pred")
    if method == "近28天均值":
        return fit.groupby("unit").tail(28).groupby("unit").qty.mean().rename("pred")
    return fit.groupby("unit").qty.apply(ses).rename("pred")


def evaluate(actual, fit, method, wt):
    pred = unit_forecast(fit, method)
    d = actual[["unit", "ts", "qty", "product"]].merge(pred, on="unit", how="inner").merge(wt, on="unit", how="left")
    err = (d.qty - d.pred).abs()
    return {
        "加权WAPE%": round((d.weight * err).sum() / (d.weight * d.qty).sum() * 100, 2),
        "加权MAE": round((d.weight * err).sum() / d.weight.sum(), 1),
        "非加权WAPE%": round(err.sum() / d.qty.sum() * 100, 2),
        "加权偏差%": round((d.weight * (d.pred - d.qty)).sum() / (d.weight * d.qty).sum() * 100, 2),
        "评估单元数": d.unit.nunique(),
    }, d


def wape(d):
    return {
        "加权WAPE%": round((d.weight * (d.qty - d.pred).abs()).sum() / (d.weight * d.qty).sum() * 100, 2),
        "加权偏差%": round((d.weight * (d.pred - d.qty)).sum() / (d.weight * d.qty).sum() * 100, 2),
        "行数": len(d),
    }


val_rows, test_rows, detail = [], [], []
horizon_rows, prod_rows, scale_rows = [], [], []

for g in GROUPS:
    tr = pd.read_csv(DATA / g / f"use_train_{g}.csv", parse_dates=["ts"])
    te = pd.read_csv(DATA / g / f"use_test_{g}.csv", parse_dates=["ts"])
    wt = pd.read_csv(DATA / g / f"weight_{g}.csv")[["unit", "weight"]]
    topo = pd.read_csv(DATA / g / f"product_topo_{g}.csv")

    val_start = tr.ts.max() - pd.Timedelta(days=VAL_DAYS - 1)
    fit, val = tr[tr.ts < val_start], tr[tr.ts >= val_start]
    print(f"[{g}] 验证集 {val_start.date()}~{tr.ts.max().date()}｜{val.ts.nunique()} 天")

    scores = {}
    for m in METHODS:
        s, _ = evaluate(val, fit, m, wt)
        val_rows.append({"组": g, "方法": m, **s})
        scores[m] = s["加权WAPE%"]
        print(f"     {m:<14} 加权WAPE {s['加权WAPE%']:>6.2f}%  偏差 {s['加权偏差%']:>+7.2f}%")

    best = min(scores, key=scores.get)
    print(f"     → 验证集最优：{best}（{scores[best]:.2f}%）")

    s_test, d = evaluate(te, tr, best, wt)
    test_rows.append({"组": g, "方法": best, **s_test})
    print(f"     测试集 {best}: 加权WAPE {s_test['加权WAPE%']:.2f}%  偏差 {s_test['加权偏差%']:+.2f}%")

    d = d.merge(topo.rename(columns={"product_level_2": "product"}), on="product", how="left")
    d["预测周"] = ((d.ts - te.ts.min()).dt.days // 7 + 1).astype(int)
    detail.append(d.assign(组=g)[["组", "unit", "ts", "qty", "pred"]])

    # 误差分解：预测周 / 商品大类 / 单元规模
    d["商品大类"] = d.product_level_1.fillna("未匹配").str[:8]
    share = d.groupby("商品大类").qty.sum() / d.qty.sum()

    for k, sub in d.groupby("预测周"):
        horizon_rows.append({"组": g, "预测周": int(k), **wape(sub)})
    for k, sub in d.groupby("商品大类"):
        prod_rows.append({"组": g, "商品大类": k, "需求占比%": round(share[k] * 100, 1), **wape(sub)})

    bucket = pd.qcut(tr.groupby("unit").qty.mean(), 3, labels=["小", "中", "大"]).rename("规模档")
    for k, sub in d.merge(bucket, on="unit").groupby("规模档", observed=True):
        scale_rows.append({"组": g, "规模档": str(k), **wape(sub), "单元数": sub.unit.nunique()})

save(pd.DataFrame(val_rows), "05_基线对比_验证集.csv")
save(pd.DataFrame(test_rows), "06_基线对比_测试集.csv")
save(pd.DataFrame(horizon_rows), "07_误差分解_预测周.csv")
save(pd.DataFrame(prod_rows), "08_误差分解_商品.csv")
save(pd.DataFrame(scale_rows), "09_误差分解_规模.csv")
save(pd.concat(detail, ignore_index=True), "10_预测明细_测试集.csv")
print("\n已写出 05~10 六张表到 outputs/")
