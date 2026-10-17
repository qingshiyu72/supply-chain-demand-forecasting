"""03 生成交付物：Excel 汇总 + 单页 HTML 看板。

只读 outputs/ 里 01、02 两步产出的 CSV，不重算任何指标。
"""

import pathlib
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

io = lambda name: pd.read_csv(OUT / name)
profile = io("00_数据画像.csv")
conc = io("01_需求集中度.csv")
yearly = io("02_年度水平.csv")
dow = io("03_周内效应.csv")
val = io("05_基线对比_验证集.csv")
test = io("06_基线对比_测试集.csv")
horizon = io("07_误差分解_预测周.csv")
prod = io("08_误差分解_商品.csv")
scale = io("09_误差分解_规模.csv")

# ---------- Excel ----------
xlsx = OUT / "需求预测分析报告.xlsx"
with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
    for sheet, df in {
        "1_数据画像": profile, "2_需求集中度": conc, "3_年度水平": yearly, "4_周内效应": dow,
        "5_基线对比_验证集": val, "6_基线对比_测试集": test,
        "7_误差_预测周": horizon, "8_误差_商品": prod, "9_误差_规模": scale,
    }.items():
        df.to_excel(w, sheet_name=sheet, index=False)
        ws = w.sheets[sheet]
        for i, col in enumerate(df.columns, 1):
            width = max(len(str(col)) * 2, *(len(str(v)) for v in df.iloc[:, i - 1].head(50))) + 2
            ws.column_dimensions[ws.cell(1, i).column_letter].width = min(width, 30)

# ---------- HTML 看板 ----------
def bar(value, best, worst, color):
    w = 4 + 96 * (1 - (value - best) / ((worst - best) or 1))
    return f'<div class="bar"><span style="width:{w:.0f}%;background:{color}"></span></div>'


rows_val = ""
for _, r in val.iterrows():
    grp = val[val["组"] == r["组"]]
    best, worst = grp["加权WAPE%"].min(), grp["加权WAPE%"].max()
    rows_val += (f'<tr><td>{r["组"]}</td><td>{r["方法"]}</td><td class="num">{r["加权WAPE%"]:.2f}%</td>'
                 f'<td>{bar(r["加权WAPE%"], best, worst, "#e08a4b")}</td>'
                 f'<td class="num">{r["加权偏差%"]:+.2f}%</td></tr>')

rows_h = ""
for g, sub in horizon.groupby("组"):
    row = "".join(f'<td class="num">{v:.2f}%</td>' for v in sub["加权WAPE%"])
    rows_h += f'<tr><td>{g}</td>{row}</tr>'

rows_s = ""
for g, sub in scale.groupby("组"):
    row = "".join(f'<td class="num">{v:.1f}%</td>' for v in sub["加权WAPE%"])
    rows_s += f'<tr><td>{g}</td>{row}</tr>'

rows_p = ""
for _, r in prod.iterrows():
    rows_p += (f'<tr><td>{r["组"]}</td><td><code>{r["商品大类"]}</code></td>'
               f'<td class="num">{r["需求占比%"]:.1f}%</td><td class="num">{r["加权WAPE%"]:.2f}%</td>'
               f'<td class="num">{r["加权偏差%"]:+.2f}%</td></tr>')

wk = "<th>组</th>" + "".join(f"<th>W{k}</th>" for k in horizon["预测周"].unique())
apex = {g: (s["加权WAPE%"].iloc[0], s["加权WAPE%"].iloc[-1]) for g, s in horizon.groupby("组")}
mult = max(b / a for a, b in apex.values())

html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>供应链需求预测 · 基线对比看板</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin:0; padding:32px 20px 60px; background:#f6f5f2; color:#23211e;
         font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif; }}
  .wrap {{ max-width:980px; margin:0 auto; }}
  h1 {{ font-size:24px; margin:0 0 6px; letter-spacing:.5px; }}
  .sub {{ color:#6f6a62; font-size:13px; margin-bottom:26px; }}
  .kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:26px; }}
  .kpi {{ background:#fff; border:1px solid #e6e2da; border-radius:10px; padding:16px 18px; }}
  .kpi .v {{ font-size:26px; font-weight:650; letter-spacing:-.5px; }}
  .kpi .l {{ font-size:12px; color:#6f6a62; margin-top:6px; line-height:1.5; }}
  .card {{ background:#fff; border:1px solid #e6e2da; border-radius:10px; padding:20px 22px; margin-bottom:20px; }}
  h2 {{ font-size:15px; margin:0 0 14px; font-weight:650; }}
  h2 span {{ font-weight:400; color:#8a8478; font-size:12px; margin-left:8px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th {{ text-align:left; font-weight:600; color:#6f6a62; font-size:12px;
        padding:6px 8px; border-bottom:1px solid #e6e2da; }}
  td {{ padding:7px 8px; border-bottom:1px solid #f1eee8; }}
  tr:last-child td {{ border-bottom:none; }}
  .num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  .bar {{ background:#f1eee8; border-radius:3px; height:8px; overflow:hidden; }}
  .bar span {{ display:block; height:8px; border-radius:3px; }}
  code {{ background:#f1eee8; padding:1px 5px; border-radius:4px; font-size:12px; }}
  ul {{ margin:0; padding-left:20px; font-size:13.5px; line-height:1.85; color:#33302b; }}
  .foot {{ color:#8a8478; font-size:12px; margin-top:24px; line-height:1.7; }}
</style></head><body><div class="wrap">
<h1>供应链需求预测 · 基线对比</h1>
<div class="sub">A / B 两条供应链｜训练 2018-06-04 ~ 2021-03-01｜测试 2021-03-02 ~ 2021-06-07（98 天）｜指标为加权 WAPE</div>

<div class="kpis">
  <div class="kpi"><div class="v">3.85%</div><div class="l">A 组测试集加权 WAPE<br>最优方法：指数平滑</div></div>
  <div class="kpi"><div class="v">4.04%</div><div class="l">B 组测试集加权 WAPE<br>最优方法：指数平滑</div></div>
  <div class="kpi"><div class="v">16%</div><div class="l">A 组单元贡献 80% 需求<br>B 组为 14.6%</div></div>
  <div class="kpi"><div class="v">{mult:.1f}×</div><div class="l">第 14 周误差 / 第 1 周误差<br>A {"{:.1f}".format(apex["A"][1] / apex["A"][0])}× / B {"{:.1f}".format(apex["B"][1] / apex["B"][0])}×</div></div>
</div>

<div class="card"><h2>方法对比<span>验证集，98 天等长回测；条越短越好</span></h2>
<table><tr><th>组</th><th>方法</th><th class="num">加权WAPE</th><th></th><th class="num">加权偏差</th></tr>
{rows_val}</table></div>

<div class="card"><h2>误差随预测步长衰减<span>最优方法，测试集逐周</span></h2>
<table><tr>{wk}</tr>{rows_h}</table></div>

<div class="card"><h2>误差按单元规模<span>小 / 中 / 大各占三分之一单元数</span></h2>
<table><tr><th>组</th><th class="num">小</th><th class="num">中</th><th class="num">大</th></tr>{rows_s}</table></div>

<div class="card"><h2>误差按商品大类</h2>
<table><tr><th>组</th><th>大类</th><th class="num">需求占比</th><th class="num">加权WAPE</th><th class="num">加权偏差</th></tr>
{rows_p}</table></div>

<div class="card"><h2>结论</h2><ul>
<li><b>历史窗口比模型复杂度重要。</b>全历史均值的加权 WAPE 为 16%/20%，偏差 -13%/-19%——2018 年需求水平是 2019 年的两倍，把三年前的数据等权平均会系统性高估。改用近 28 天均值即降到 3.7%/6.5%。</li>
<li><b>周内没有可利用的规律。</b>七天相对指数全部落在 0.999~1.001，所以预测采用单元级常数，不做周内画像。</li>
<li><b>误差几乎全部来自少数大单元之外的地方。</b>大单元加权 WAPE 约 3.8%，小单元高达 27%，且偏差为 -22%——低量单元的需求在上升，平滑法追不上趋势。</li>
<li><b>98 天跨度内的误差累积是主要的改进空间。</b>误差从第 1 周的 1.1%~1.3% 升到第 14 周的 6% 左右，说明常数预测抓不住趋势，需要引入趋势项或缩短重估周期。</li>
</ul></div>

<div class="foot">数据来源：天池「供应链需求预测」公开数据集（单元、地理、商品 ID 均已脱敏）。<br>
权重视角：weight_X.csv 提供的单元权重已用于全部指标计算，因此结论偏向业务量大的单元。</div>
</div></body></html>"""

(OUT / "需求预测看板.html").write_text(html, encoding="utf-8")
print(f"已生成：{xlsx.name}、需求预测看板.html")
