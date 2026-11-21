"""
Used car prices: should the dealer buy this car?

Reproduces every number and chart for the slides.
Run:  python analysis.py
Outputs are written to this folder (results.txt, charts as PNG, CSVs of predictions and auction decisions).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
PREP_COST = 1500
FEATURES = ["Mileage_K", "Age_Years", "Horsepower", "Accident", "Num_Owners"]

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


def r2(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


# ---------------------------------------------------------------- 1. Data
sales = pd.read_csv("used_car_sales.csv")
auction = pd.read_csv("auction_lots.csv")
for df in (sales, auction):
    df["Accident"] = (df["Accident"].str.lower() == "yes").astype(int)

log("=== DATA ===")
log(f"used_car_sales.csv: {len(sales)} cars sold in the last year")
log(f"auction_lots.csv: {len(auction)} auction lots (held out, used only to test the decision rule)")

test = sales.sample(frac=0.30, random_state=SEED)
train = sales.drop(test.index)
log(f"Random 70/30 split (seed={SEED}): train = {len(train)}, test = {len(test)}")
log()

# ---------------------------------------------------------------- 2. Fit
def fit(cols):
    X = sm.add_constant(train[cols])
    return sm.OLS(train["Sale_Price"], X).fit()


full = fit(FEATURES)
log("=== MODEL A: all five features (training cars) ===")
log(full.summary().as_text())
log()

# Drop any feature that is not significant at 5%, refit, compare adjusted R^2
weak = [c for c in FEATURES if full.pvalues[c] > 0.05]
log(f"Features with p > 0.05: {weak if weak else 'none'}")

if weak:
    kept = [c for c in FEATURES if c not in weak]
    reduced = fit(kept)
    log()
    log(f"=== MODEL B: without {', '.join(weak)} ===")
    log(reduced.summary().as_text())
    log()
    log(f"Adjusted R^2  all features: {full.rsquared_adj:.4f}   reduced: {reduced.rsquared_adj:.4f}")
    # Hand over the simpler model unless dropping costs adjusted R^2
    if reduced.rsquared_adj >= full.rsquared_adj - 0.001:
        v1, v1_cols, dropped = reduced, kept, weak
    else:
        v1, v1_cols, dropped = full, FEATURES, []
else:
    v1, v1_cols, dropped = full, FEATURES, []

log()
log("=== v1 (model handed over) ===")
log(f"Features: {v1_cols}")
log(f"Dropped: {dropped if dropped else 'none'}")
log(f"Training R^2: {v1.rsquared:.4f}   Adjusted R^2: {v1.rsquared_adj:.4f}")
log("Coefficients (dollars of sale price per unit), p-values:")
for name in v1.params.index:
    log(f"  {name:<12} {v1.params[name]:>12,.1f}   p = {v1.pvalues[name]:.2g}")
log()

# ---------------------------------------------------------------- 3. Test fit
baseline_price = train["Sale_Price"].mean()
test_base = np.full(len(test), baseline_price)
test_v1 = v1.predict(sm.add_constant(test[v1_cols], has_constant="add"))

log("=== FIT ON THE 72 TEST CARS (never seen in fitting) ===")
log(f"Baseline prediction (training average): ${baseline_price:,.0f} for every car")
log(f"{'':<10}{'R^2':>8}{'RMSE ($)':>12}")
log(f"{'Baseline':<10}{r2(test.Sale_Price, test_base):>8.3f}{rmse(test.Sale_Price, test_base):>12,.0f}")
log(f"{'v1':<10}{r2(test.Sale_Price, test_v1):>8.3f}{rmse(test.Sale_Price, test_v1):>12,.0f}")
log()

pd.DataFrame({
    "Actual_Sale_Price": test["Sale_Price"],
    "Baseline_Pred": test_base,
    "v1_Pred": test_v1.round(0),
}).to_csv("test_predictions.csv")

# ---------------------------------------------------------------- 4. Decision
def predict_price(car: dict) -> float:
    """v1 predicted lot sale price for one car (features as in the CSVs)."""
    c = dict(car)
    if isinstance(c.get("Accident"), str):
        c["Accident"] = 1 if c["Accident"].lower() == "yes" else 0
    return float(v1.params["const"] + sum(v1.params[f] * c[f] for f in v1_cols))


def decide(car: dict, auction_price: float) -> str:
    """Dealer rule: buy if auction price + $1,500 prep is below v1's predicted sale price."""
    return "buy" if auction_price + PREP_COST < predict_price(car) else "pass"


def decide_baseline(car: dict, auction_price: float) -> str:
    """Baseline rule: buy if auction price + $1,500 is below the average sale price."""
    return "buy" if auction_price + PREP_COST < baseline_price else "pass"


auction["v1_Pred"] = [predict_price(r) for r in auction[FEATURES].to_dict("records")]
auction["Baseline_Decision"] = [decide_baseline(r, p) for r, p in
                                zip(auction[FEATURES].to_dict("records"), auction["Auction_Price"])]
auction["v1_Decision"] = [decide(r, p) for r, p in
                          zip(auction[FEATURES].to_dict("records"), auction["Auction_Price"])]
auction["Profit_If_Bought"] = auction["Later_Sale_Price"] - auction["Auction_Price"] - PREP_COST
auction.round(0).to_csv("auction_decisions.csv", index=False)

log("=== DECISION RULE ON THE 40 AUCTION LOTS ===")
summary = {}
for name, col in [("Baseline", "Baseline_Decision"), ("v1", "v1_Decision")]:
    bought = auction[auction[col] == "buy"]
    summary[name] = dict(
        bought=len(bought),
        losers=int((bought["Profit_If_Bought"] < 0).sum()),
        profit=int(bought["Profit_If_Bought"].sum()),
    )
log(f"{'':<10}{'Bought':>8}{'Lost money':>12}{'Total profit ($)':>18}")
for name, s in summary.items():
    log(f"{name:<10}{s['bought']:>8}{s['losers']:>12}{s['profit']:>18,}")
diff = summary["v1"]["profit"] - summary["Baseline"]["profit"]
log(f"Difference in profit (v1 - baseline): ${diff:,}")
log()
log("Lots bought by v1: " + ", ".join(str(l) for l in auction.loc[auction.v1_Decision == "buy", "Lot"]))
log("Lots bought by baseline: " + ", ".join(str(l) for l in auction.loc[auction.Baseline_Decision == "buy", "Lot"]))
log()
lot16 = auction.iloc[15]
log(f"Example: decide(lot 16 features, ${lot16['Auction_Price']:,}) -> "
    f"{decide(lot16[FEATURES].to_dict(), lot16['Auction_Price'])}")

# ---------------------------------------------------------------- Charts
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})

fig, ax = plt.subplots(figsize=(7, 6))
lo, hi = 0, 50000
ax.plot([lo, hi], [lo, hi], color="#999999", lw=1, ls="--", label="Perfect prediction")
ax.scatter(test["Sale_Price"], test_base, s=28, color="#B0B0B0", label=f"Baseline (R² {r2(test.Sale_Price, test_base):.2f})")
ax.scatter(test["Sale_Price"], test_v1, s=28, color="#1F5AA6", label=f"v1 model (R² {r2(test.Sale_Price, test_v1):.2f})")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("Actual sale price ($)")
ax.set_ylabel("Predicted sale price ($)")
ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
ax.set_title("Predicted vs actual sale price, 72 test cars")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig("pred_vs_actual.png", dpi=200)

fig, ax = plt.subplots(figsize=(7, 4.5))
names = list(summary)
profits = [summary[n]["profit"] for n in names]
bars = ax.bar(names, profits, color=["#B0B0B0", "#1F5AA6"], width=0.5)
for b, n in zip(bars, names):
    s = summary[n]
    money = f"-${-s['profit']:,}" if s['profit'] < 0 else f"${s['profit']:,}"
    ax.annotate(f"{money}\n{s['bought']} bought, {s['losers']} lost money",
                (b.get_x() + b.get_width() / 2, b.get_height()),
                ha="center", va="bottom" if b.get_height() >= 0 else "top", fontsize=11)
ax.axhline(0, color="black", lw=0.8)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"-${-x:,.0f}" if x < 0 else f"${x:,.0f}"))
ax.set_ylabel("Total profit on 40 auction lots")
ax.set_title("Profit from each buying rule")
ax.margins(y=0.25)
fig.tight_layout()
fig.savefig("auction_profit.png", dpi=200)

with open("results.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
