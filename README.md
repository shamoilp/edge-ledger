# edge-ledger

Small, tested tools for one question: **is this stock-screening factor an edge, or noise?**

They came out of six months of trying to validate my own swing-trading screener across three
markets (India NSE, US, Saudi Tadawul). I registered every factor test in a ledger and ran each
one through the checks below. This README is the write-up of what that taught me. The package
is the part of the method that other people can reuse.

```
edge-ledger ic          per-date rank IC, with a t-stat that survives overlapping returns
edge-ledger permute     within-date permutation test + date bootstrap
edge-ledger walkforward purged, anchored walk-forward of the IC (and of factor *selection*)
edge-ledger ledger      register tests; the adoption bar rises with every one you add
```

`numpy` + `scipy`, ~1,000 lines, 73 tests, no market-data dependency.

---

## What 70 registered tests on a retail screener taught me

### The short version

- **36 of 70 tests are dead, and none has passed every check.** 8 are on watch, 18 are
  observational, 4 are queued. 4 are marked adopted, but those are parts of a ranking score that
  later failed its own test, plus one variant adopted to remove a factor that was hurting, not
  to add one that helps.
- **The 50-trade paper validation found no edge.** Win rate 48% (24W / 26L), Z = −0.28 against a
  1.96 bar. Gross P&L was positive but not significant. After round-trip costs (0.35%) and
  opening-gap slippage, total P&L turned negative and the win rate fell to 44%.
- **Most of the useful lessons were about measurement, not factors.** Several of my "findings"
  were artifacts of how returns were computed or how t-stats were counted. Those are listed
  under [What fooled me](#what-fooled-me).
- **The binding constraint is sample size.** A per-date IC t-stat is IR × √n. With 8 dates, a
  factor needs an IR above 1.16 to clear a 3.3 bar, which almost nothing has. With 40 dates the
  bar drops to 0.52. More ideas do not help until n grows.

I started the ledger part-way through: the first 38 entries were back-filled from tests I had
already run. Every test after that was registered before its result was looked at.

### What died

| Idea | What it looked like first | What happened |
|---|---|---|
| Checklist pass count (how many of 8 screen checks passed) | The obvious ranking key | Spearman with forward 20-day R-multiple: **+0.037** (n = 640). No relationship. |
| Structural composite (stage, trend, OBV, sector rotation, hand-weighted) | Top quartile median R +0.110 vs bottom −0.020. Filtering on it lifted win rate 55% → 69% | Within-date permutation: p = 0.18, the real IC sat at the 82nd percentile of luck. Later 20-day IC: India +0.015, US −0.107, Saudi −0.064. Purged walk-forward pooled out-of-sample: −0.007. |
| Relative strength | India 20-day IC +0.150 (t +2.90) on the first sample | A pre-registered prediction about its term structure failed. US 20-day IC **−0.450** and Saudi **−0.308**: significantly *anti*-predictive. India decayed to +0.022. |
| ADX (trend strength) | India 20-day IC +0.261 early, +0.104 (t ≈ 3.0) later | Never cleared the bar. Saudi −0.117 with a 95% CI that excludes zero. Purged walk-forward pooled out-of-sample: −0.026. |
| Scan recurrence (how often a name reappears) | +0.198 (t 2.78) on the first sample | +0.044 (t 1.46) once more dates matured. |
| Day-0 strength | India 20-day IC +0.167, pooled t +6.69. The only factor to clear the bar | Only one market had matured 20-day data. At 10 days, where all three do, the sign flips in two of them. Purged walk-forward: in-sample ~+0.17 → out-of-sample +0.06 (efficiency 0.34), falling fold by fold. |
| 252-day volatility rank | 5-day pooled t +3.21, same sign in all three markets | Walk-forward efficiency 0.08. |
| Trailing stops and 3-tranche scale-out | Standard advice | Holding to the 20-day time stop beat both in every regime cohort tested. Trailing cut winners early and did not reduce drawdown. |
| Correlation-fragility regime score | Rose sharply into 2018, March 2020 and 2022 in all three markets | Over 12 years, its correlation with forward drawdown has a CI straddling zero in every market. The "crisis" band had the **highest** median 60-day forward return in all three (+4.2% / +8.7% / +3.1%). Using it to cut exposure would have de-risked into the best periods. |
| Live entry gate (4-of-6 market-weather reconstruction) | Common-sense "don't trade in a storm" | Over 438 weeks of 2018–2026 India replay, the 20-day forward return of signals in weeks it allowed averaged +1.60%, vs +2.23% in weeks it blocked, and the sign was negative in both halves of the sample. (It still has a job as a discipline rule, just not as a return filter.) |
| Post-capitulation break-up (low-float names) | Explosion rate 30.2% vs a 5.6% base | A matched control on the same symbols reached 22.9%. Compared date by date, the signal beat the control on 13–17 of 27 dates: a coin flip. Median 5-day return was negative in all 12 grid cells. |

The pattern across the table: a strong first read on a small or single-regime sample, then
decay, a sign flip in another market, or a control that explains most of the effect.

### What fooled me

These are the mistakes worth copying the tests for. Each one made a result look better or
worse than it was, and none of them raised an error.

**1. Stock splits in the price data.** Vendor price series are not consistently split-adjusted,
and the adjustment state of a symbol changes over time.
- A 1:125 reverse split logged a 5-day maximum gain of **+11,724%** against a true **+1.4%**.
  32 of the 35 rows with gains above 1,000% in one corpus sat across a reverse split.
- A 4:1 forward split turned a 20-day return of **+15.9%** into **−71.0%**: a sign flip, which is
  what corrupts an IC.
- One paper book's mean trade went from **+58.0% to −7.6%** once fixed. Its single "jackpot"
  trade, +2,123%, was really −11.6%.
- The same split also removed the stock from the screen: the reverse split moved its price to
  ~10,000% above its own 52-week low, so the "near the low" filter rejected it.

There are two failure modes, and a fix for one breaks the other: a **raw** series has a real
jump at the split; a **back-adjusted** series has no jump, but any price you *stored* earlier is
now on the old scale. What works: find the split bar from the price path (the label can be a
bar off), report no split when there is no jump, and take entry and exit prices from **one**
fetch. `edge_ledger.splits` implements this, with tests for each case.

**2. Overlapping returns inflate the t-stat.** A 20-day forward return logged every day shares
19 days with the next row, so "255 dates" is nothing like 255 observations. In the synthetic
demo below, a pure-noise factor scores **t = −3.75** on all dates and **−1.37** on every 5th
date. Only the second number is honest. Every verdict in this package uses the thinned sample.

**3. Small samples need a t distribution.** Three dates with ICs of 0.085, 0.080 and 0.088 gave
**t = +36.9**, which "cleared" the normal-distribution bar of 3.38 by a factor of ten. With 2
degrees of freedom the correct Bonferroni bar is **37.4**. The package switches to t(n−1)
automatically and refuses to report a t-stat from fewer than 3 dates.

**4. Thinning in the wrong order.** "Take every 4th week" was implemented as every 4th entry of a
dict whose insertion order was by *symbol*, not date. The result was an arbitrary 25% slice
that still overlapped. Fixing it moved one gauge's correlation from +0.307 to +0.211 and
another's from −0.087 to −0.188. Found only because a second script, written separately,
disagreed.

**5. Walk-forward without a purge leaks.** If the in-sample window ends on day *t* and
out-of-sample starts on day *t+1*, the last 20 in-sample labels look into the out-of-sample
period. The purge here drops in-sample dates within `ceil(H × 7/5) + 6` calendar days of the
first out-of-sample date (34 days for a 20-day horizon). It costs 20–23 in-sample dates per
fold, which is why my US and Saudi journals produce at most one 20-day fold so far.

**6. Pooled results hide composition.** The break-up signal in the table beat its controls by
3–11 points pooled, depending on the grid cell. Paired date by date, the advantage disappeared. When some dates contribute
many signal rows and others few, a pooled gap can come entirely from *which dates* are in the
sample. Compare within dates.

**7. Multiplicity.** Seventy tests at a 5% level give about 3.5 false positives for free. The
ledger sets the bar from the number of registered tests (two-sided Bonferroni, as in Harvey,
Liu & Zhu 2016): 1.96 for one test, 3.38 for seventy. Picking the best of four variants on a
small sample is the same problem in miniature; `walkforward --select` measures it.

**8. A green exit code is not a health check.** A nightly job that filled forward returns
printed "D20 0" every night for a month while exiting 0, because it was filling the shorter
horizons. Its backlog grew from 3,756 to 4,120 rows before anyone noticed. A separate alert
path never delivered a single message because of an import error it caught and logged. The
fix in both cases was a check on the output data, not on the process.

---

## Method

1. **Register before you look.** `edge-ledger ledger add <name>` before running the test. Never
   delete entries, including failures; N is what keeps the bar honest.
2. **Per-date cross-sectional rank IC.** Spearman correlation between the factor and the forward
   return across names, one value per date. Ties get their average rank, so a discrete score
   does not depend on row order.
3. **t-stat on non-overlapping dates**, judged against a Bonferroni bar at the ledger's N, with
   t(n−1) critical values at small n.
4. **Within-date permutation.** Shuffle the factor across names inside each date, many times,
   and see where the real mean IC falls. Plus a bootstrap over dates for a confidence interval.
5. **Purged, anchored walk-forward.** Fit nothing; just check that the in-sample sign and size
   carry out of sample. Reports out-of-sample sign retention, efficiency (OOS IC / IS IC), the
   overfit ratio (IS IR / OOS IR) and decay per fold. `--select` re-picks the best factor on
   each in-sample window and scores only that pick, which tests the decision you actually make.
6. **Sign consistency across markets.** Not built into the package, but it was my promotion rule:
   pooled |t| above the bar *and* the same sign in at least two markets.

## Quickstart

```bash
git clone https://github.com/shamoilp/edge-ledger
cd edge-ledger
pip install -e ".[test]"
python examples/make_synthetic.py      # seeded journal with known answers
pytest -q
```

The synthetic journal has 30 names over a year of business days. `momentum_like` is truly
predictive all year, `decays` is predictive only in the first half, and `noise_01..10` have no
link to returns. You know the answers before you run the tools, so you can check that the
tools agree.

```text
$ edge-ledger ic examples/synthetic.csv --factor momentum_like decays noise_01 noise_07 --returns fwd_5

  Per-date IC · returns fwd_5 (H = 5) · N = 4 (no --ledger: only this command's tests counted, your real N is higher)

  factor             dates  mean IC     IR  t(all)  | thin n  t(thin)    bar  verdict
  momentum_like        255   +0.077  +0.45   +7.17  |     51    +3.48   2.59  CLEARS
  decays               255   +0.034  +0.18   +2.82  |     51    +1.54   2.59  does not clear
  noise_01             255   -0.044  -0.23   -3.75  |     51    -1.37   2.59  does not clear
  noise_07             255   +0.021  +0.11   +1.80  |     51    +0.99   2.59  does not clear
```

Note `noise_01`: on all dates it would pass, thinned it does not.

```text
$ edge-ledger permute examples/synthetic.csv --factor noise_07 --returns fwd_5

  Permutation test · noise_07 vs fwd_5 · every 5th date · N = 1 (no --ledger: ...)
  dates 51 · real mean IC +0.0285
  luck: mean -0.0001, sd 0.0261 · real IC sits at the 86.2th percentile of luck
  p = 0.1378 (positive direction) → not distinguishable from luck
  date bootstrap 90% CI [-0.0192, +0.0747] → straddles 0
```

```text
$ edge-ledger walkforward examples/synthetic.csv --factor decays --returns fwd_5

  255 dates · purge 13 calendar days + 2% of IS · 24 folds, 0 skipped
  ...
  Rules of thumb (WF03 heuristics, not tests)
    overfit_ratio   +3.58  [red] > 2.5: likely overfit
    efficiency      +0.39  [red] < 0.4: poor transfer
    sign held       16/24  folds where OOS IC kept the IS sign

  every 5th date: n 47, t +1.17, bar 2.01 → does not clear
```

`decays` looks strong in-sample early on and fails to transfer, which is what it was built to do.

```text
$ edge-ledger walkforward examples/synthetic.csv --select momentum_like decays noise_01 noise_07 --returns fwd_5
  ...
  every 5th date: n 47, t +2.84, bar 2.60 → CLEARS
  average candidate OOS IC +0.022 → selection skill +0.045
```

```text
$ edge-ledger ledger add momentum_like --source "examples"
$ edge-ledger ledger bar --n-obs 3

  1 test registered → adoption bar |t| >= 4.30 (two-sided Bonferroni, FWER 5%, t(2) for 3 dates)
```

Pass `--ledger edge_ledger.json` to `ic`, `permute` and `walkforward` so N comes from your
ledger. Without it, N counts only the factors in that one command, and the output says so.

## Journal format

One CSV row per (date, symbol):

| column | type | notes |
|---|---|---|
| `date` | `YYYY-MM-DD` | required |
| `symbol` | string | required; one row per date and symbol |
| any factor | number | blank, `nan` or `inf` = missing |
| forward returns | number, any unit | blank = not yet matured. The horizon is read from the digits in the name (`fwd_20` → 20 days) or given with `--horizon-days` |

Dates with fewer than 5 usable names are skipped (`--min-names`). Forward returns should be
computed from one consistent price fetch; see `edge_ledger.splits`.

## Limitations

- The findings above come from about six months of live daily journals plus a 2018–2026 replay
  for some tests. That is short, and the replay universe has survivorship bias.
- Much of the live window overlapped one volatile period, so several early results were
  measured in a single regime.
- Paper trades only. Nothing here is investment advice, and the screener itself is not included.
- Bonferroni is conservative, and the permutation test is anti-conservative if you run it on
  overlapping dates (`--all-dates`). The defaults avoid the second problem and accept the first.

## Credits

- Walk-forward fold generation, purge and the WF03 heuristics are adapted from the
  [QuantJourney Backtester](https://github.com/QuantJourneyOrg/quantjourney-bt) (Apache-2.0); see `NOTICE`.
- Harvey, Liu & Zhu (2016), "…and the Cross-Section of Expected Returns", for the multiplicity bar.
- Chan, *Algorithmic Trading* (2013), Ch. 1, for backtest randomization.
- López de Prado, *Advances in Financial Machine Learning* (2018), Ch. 7, for purging.
- Pardo, *The Evaluation and Optimization of Trading Strategies* (2008), for walk-forward discipline.

Apache-2.0. See `LICENSE`.
