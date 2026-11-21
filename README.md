# Used car prices: should the dealer buy this car?

Linear regression of used car sale price on mileage, age, horsepower, accident history and number of owners, used to decide which cars to buy at wholesale auction.

## Run it

```bash
pip install -r requirements.txt
python analysis.py
```

Every number and chart on the slides is written to `output/` by that one script.

## Files

| Path | What it is |
|---|---|
| `analysis.py` | The full analysis: data split, model fitting, test-set evaluation, decision rule, charts |
| `data/used_car_sales.csv` | 240 cars sold on the lot in the last year (used to fit and test the model) |
| `data/auction_lots.csv` | 40 auction lots (used only to test the buying rule) |
| `output/results.txt` | Full regression output and every number on the slides |
| `output/pred_vs_actual.png` | Predicted vs actual sale price on the 72 test cars, baseline vs v1 |
| `output/auction_profit.png` | Total profit on the 40 auction lots, baseline rule vs v1 rule |
| `output/test_predictions.csv` | Baseline and v1 prediction for each test car |
| `output/auction_decisions.csv` | Buy/pass from each rule and profit for every auction lot |

## Method

1. **Split:** random 70/30 split of the 240 cars (seed 42): 168 training, 72 test.
2. **Fit:** OLS on the training cars with all five features. `Num_Owners` was not significant (p = 0.87), so it was dropped and the model refitted. Adjusted R² rose slightly (0.869 to 0.870), so the four-feature model is v1.
3. **Evaluate:** baseline (training average price for every car) vs v1 on the 72 test cars, using R² and RMSE.
4. **Decide:** `decide(car, auction_price)` returns `"buy"` if auction price + $1,500 prep cost is below v1's predicted sale price, otherwise `"pass"`. Applied with the baseline rule to the 40 auction lots; profit per bought car = `Later_Sale_Price - Auction_Price - 1,500`.

## Key results

| | R² (test) | RMSE (test) |
|---|---|---|
| Baseline | -0.01 | $6,494 |
| v1 | 0.82 | $2,716 |

| Rule | Cars bought | Lost money | Total profit |
|---|---|---|---|
| Baseline | 14 | 10 | -$6,000 |
| v1 | 17 | 6 | $15,800 |
