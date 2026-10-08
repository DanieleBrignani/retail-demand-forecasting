> **SUPERSEDED by [`results_weekly_v2.md`](results_weekly_v2.md) (protocol `weekly-v2`).** Original protocol: [`weekly_protocol.md`](weekly_protocol.md); the v1 code is not kept in this repository. Known defects of v1: all 9,414 series scored at every origin including products not yet launched (no universe policy);
> no segment, fallback, origin-variability, crossing or active-series coverage diagnostics; interval crossing repaired silently. It never used prices, so the price defect does not affect it. Its verdict (not a serving candidate) is retained.
> **Also affected by the unit-overflow defect** (`int16`, 131,072 units lost; [`audit_int16_overflow.md`](audit_int16_overflow.md)): all numbers below were computed on the corrupted panel.

# Weekly study — results (protocol: [`weekly_protocol.md`](weekly_protocol.md), fixed before the run)

> **Provenance.** Verbatim output of `retailfc weekly backtest` (run 2026-10-05, dataset `retail-369d316281ea`, 105 complete weeks, 9,414 product×channel series, 10 rolling origins 65…101, horizons 1–4 weeks).
> Not a pristine holdout (see protocol header). Recipe not tuned. Weekly zero share: 75.7% (daily: 91.7%).

## Verdict against the pre-registered criteria: **`lgbm_w` is NOT a serving candidate**
| criterion | result | met |
|---|---|---|
| 1. MASE ≥ 5% better than best baseline and better in ≥ 7/10 origins | MASE 3.079 vs best baseline `mean4` 2.336; better in **0/10** origins | **no** |
| 2. pooled WAPE ≤ 0.90 | 0.834 | yes |
| 3. pooled |bias| ≤ 10% and L3 |bias| ≤ 10% in ≥ 6/10 origins | −7.8%; 6/10 | yes (narrowly) |
| 4. pinball below `empirical_q` and 80% coverage in [0.70, 0.90] | pinball 3.137 vs 3.531 (better); coverage **0.915** (too wide) | **no** |

## What the numbers say (interpretation, not part of the pre-registration)
* Aggregating to weeks helped the *problem* (WAPE of the best models ≈ 0.83–0.85 instead of ≥ 1.04 daily, and the grand-total level is forecast with WAPE 16%), but **the global model is only about 2% better than a 4-week moving average on pooled WAPE (0.834 vs 0.8505) and 3% on RMSE.** That is not a strong result.
* `lgbm_w` is best on volume-weighted metrics (WAPE, RMSE, pinball, WAPE at channel/total level: 0.196/0.161 vs 0.226/0.182 for `mean4`) and **worst-but-one on MASE**, a series-averaged scaled metric dominated by small intermittent series (and the all-zero forecast has MASE 2.36, better than most methods). It serves big series well and small intermittent series poorly.
* Bias: −7.8% pooled; the first four origins (−15% … −25%, all in the Dec–Mar low season after a first year with a single Christmas) drive it. Croston/SBA over-forecasts (+33%); `seasonal_naive52` is poor (one prior year only).
* The quantile forecasts are calibrated conservatively (coverage 0.915 for a nominal 0.80; no conformal step) but are sharper in pinball loss than empirical quantiles.
* Nothing here supports a claim of significance, and the daily study's conclusions are unchanged.

## Raw output
```text
== point
                    wape      mae      rmse  bias_pct    mase   rmsse
model                                                                
all_zero          1.0000  11.5420   61.5671   -1.0000  2.3575  5.9893
naive             0.9811  11.3243   59.5964   -0.0393  2.5316  5.8518
mean4             0.8505   9.8162   48.6251   -0.0944  2.3357  5.8801
mean13            0.8550   9.8681   48.2644   -0.1560  2.3634  5.9254
seasonal_naive52  1.3336  15.3923  102.7234    0.0694  2.8200  6.0259
croston_sba       1.3006  15.0119   63.1334    0.3294  4.4352  6.0251
tsb               0.9070  10.4688   49.8488   -0.0824  2.4119  5.9260
lgbm_w            0.8340   9.6259   46.9237   -0.0781  3.0786  5.9614

== per_origin
               model  origin    wape    mase  bias_pct  n_series_scored
0           all_zero      65  1.0000  0.7873   -1.0000             7067
1              naive      65  0.9818  0.8669   -0.0504             7067
2              mean4      65  0.8912  0.8211   -0.0765             7067
3             mean13      65  1.0718  0.9573    0.1748             7067
4   seasonal_naive52      65  1.7408  1.3227    0.5386             7067
5        croston_sba      65  1.6690  2.6931    0.8349             7067
6                tsb      65  1.1303  1.0055    0.2903             7067
7             lgbm_w      65  0.8535  1.1894   -0.1899             7067
8           all_zero      69  1.0000  1.0469   -1.0000             7166
9              naive      69  1.0577  1.2321   -0.0346             7166
10             mean4      69  0.9305  1.1445   -0.0738             7166
11            mean13      69  0.9305  1.1453   -0.1527             7166
12  seasonal_naive52      69  1.3205  1.4127   -0.0252             7166
13       croston_sba      69  1.5918  3.0424    0.6340             7166
14               tsb      69  1.0807  1.2482    0.1328             7166
15            lgbm_w      69  0.8771  1.3519   -0.2497             7166
16          all_zero      73  1.0000  2.3155   -1.0000             7230
17             naive      73  0.9409  2.1605   -0.4419             7230
18             mean4      73  0.9036  2.3071   -0.0945             7230
19            mean13      73  0.8712  2.3886   -0.1707             7230
20  seasonal_naive52      73  1.4000  2.8157    0.1034             7230
21       croston_sba      73  1.4505  4.3795    0.4803             7230
22               tsb      73  0.9684  2.4584   -0.0167             7230
23            lgbm_w      73  0.8315  2.4512   -0.2213             7230
24          all_zero      77  1.0000  1.4535   -1.0000             7402
25             naive      77  1.2220  2.1131    0.2901             7402
26             mean4      77  0.9542  1.5739    0.0693             7402
27            mean13      77  0.9299  1.5124   -0.0316             7402
28  seasonal_naive52      77  1.3132  1.8263    0.0192             7402
29       croston_sba      77  1.5544  3.6328    0.6735             7402
30               tsb      77  1.0298  1.5696    0.0836             7402
31            lgbm_w      77  0.8471  1.8976   -0.1543             7402
32          all_zero      81  1.0000  2.3124   -1.0000             7571
33             naive      81  1.2534  2.7934    0.3409             7571
34             mean4      81  0.9594  2.3875    0.0543             7571
35            mean13      81  0.9551  2.3813    0.0612             7571
36  seasonal_naive52      81  1.3811  2.7111    0.0474             7571
37       croston_sba      81  1.5668  4.5441    0.7243             7571
38               tsb      81  1.0418  2.4209    0.1531             7571
39            lgbm_w      81  0.9468  2.7820    0.0752             7571
40          all_zero      85  1.0000  2.4635   -1.0000             7685
41             naive      85  0.9188  2.8000   -0.2130             7685
42             mean4      85  0.8305  2.4666   -0.2142             7685
43            mean13      85  0.8333  2.4825   -0.1927             7685
44  seasonal_naive52      85  1.3798  2.8380   -0.0125             7685
45       croston_sba      85  1.3015  4.6104    0.3237             7685
46               tsb      85  0.8953  2.5243   -0.1100             7685
47            lgbm_w      85  0.8807  3.7177   -0.0846             7685
48          all_zero      89  1.0000  2.0811   -1.0000             7788
49             naive      89  1.0545  2.1571   -0.0338             7788
50             mean4      89  0.9068  2.1044    0.0471             7788
51            mean13      89  0.8467  2.1193   -0.0716             7788
52  seasonal_naive52      89  1.2767  2.5129   -0.0572             7788
53       croston_sba      89  1.3046  4.2134    0.4366             7788
54               tsb      89  0.8748  2.1371   -0.0233             7788
55            lgbm_w      89  0.8155  2.4501   -0.0872             7788
56          all_zero      93  1.0000  4.9122   -1.0000             7927
57             naive      93  0.8384  4.6144   -0.2944             7927
58             mean4      93  0.7915  4.6529   -0.3337             7927
59            mean13      93  0.8129  4.8172   -0.3550             7927
60  seasonal_naive52      93  1.3289  5.5125    0.0814             7927
61       croston_sba      93  1.0969  6.9269   -0.0471             7927
62               tsb      93  0.8175  4.8467   -0.3353             7927
63            lgbm_w      93  0.8801  6.5892   -0.0351             7927
64          all_zero      97  1.0000  3.2151   -1.0000             8156
65             naive      97  1.0457  3.7267    0.1752             8156
66             mean4      97  0.8494  3.2015    0.0124             8156
67            mean13      97  0.8177  3.0960   -0.2208             8156
68  seasonal_naive52      97  1.2542  3.7606    0.0024             8156
69       croston_sba      97  1.1246  5.2778    0.0911             8156
70               tsb      97  0.8362  3.1371   -0.2100             8156
71            lgbm_w      97  0.8539  4.9816    0.0522             8156
72          all_zero     101  1.0000  2.5883   -1.0000             8294
73             naive     101  0.7821  2.4676   -0.0273             8294
74             mean4     101  0.6952  2.3432   -0.1573             8294
75            mean13     101  0.7155  2.3881   -0.2610             8294
76  seasonal_naive52     101  1.1884  3.0751    0.0910             8294
77       croston_sba     101  0.9892  4.6240   -0.0093             8294
78               tsb     101  0.7307  2.4322   -0.2706             8294
79            lgbm_w     101  0.6735  2.8232   -0.0429             8294

== levels
                               wape  bias_pct
model            level                       
all_zero         L2_channel  1.0000   -1.0000
                 L3_total    1.0000   -1.0000
croston_sba      L2_channel  0.4485    0.3294
                 L3_total    0.3670    0.3294
lgbm_w           L2_channel  0.1962   -0.0781
                 L3_total    0.1611   -0.0781
mean13           L2_channel  0.2575   -0.1560
                 L3_total    0.2329   -0.1560
mean4            L2_channel  0.2257   -0.0944
                 L3_total    0.1815   -0.0944
naive            L2_channel  0.2763   -0.0393
                 L3_total    0.2146   -0.0393
seasonal_naive52 L2_channel  0.2817    0.0694
                 L3_total    0.2336    0.0694
tsb              L2_channel  0.2563   -0.0824
                 L3_total    0.2258   -0.0824

== l3_bias_by_origin
                model  origin  bias_pct
1            all_zero      65   -1.0000
3               naive      65   -0.0504
5               mean4      65   -0.0765
7              mean13      65    0.1748
9    seasonal_naive52      65    0.5386
11        croston_sba      65    0.8349
13                tsb      65    0.2903
15             lgbm_w      65   -0.1899
17           all_zero      69   -1.0000
19              naive      69   -0.0346
21              mean4      69   -0.0738
23             mean13      69   -0.1527
25   seasonal_naive52      69   -0.0252
27        croston_sba      69    0.6340
29                tsb      69    0.1328
31             lgbm_w      69   -0.2497
33           all_zero      73   -1.0000
35              naive      73   -0.4419
37              mean4      73   -0.0945
39             mean13      73   -0.1707
41   seasonal_naive52      73    0.1034
43        croston_sba      73    0.4803
45                tsb      73   -0.0167
47             lgbm_w      73   -0.2213
49           all_zero      77   -1.0000
51              naive      77    0.2901
53              mean4      77    0.0693
55             mean13      77   -0.0316
57   seasonal_naive52      77    0.0192
59        croston_sba      77    0.6735
61                tsb      77    0.0836
63             lgbm_w      77   -0.1543
65           all_zero      81   -1.0000
67              naive      81    0.3409
69              mean4      81    0.0543
71             mean13      81    0.0612
73   seasonal_naive52      81    0.0474
75        croston_sba      81    0.7243
77                tsb      81    0.1531
79             lgbm_w      81    0.0752
81           all_zero      85   -1.0000
83              naive      85   -0.2130
85              mean4      85   -0.2142
87             mean13      85   -0.1927
89   seasonal_naive52      85   -0.0125
91        croston_sba      85    0.3237
93                tsb      85   -0.1100
95             lgbm_w      85   -0.0846
97           all_zero      89   -1.0000
99              naive      89   -0.0338
101             mean4      89    0.0471
103            mean13      89   -0.0716
105  seasonal_naive52      89   -0.0572
107       croston_sba      89    0.4366
109               tsb      89   -0.0233
111            lgbm_w      89   -0.0872
113          all_zero      93   -1.0000
115             naive      93   -0.2944
117             mean4      93   -0.3337
119            mean13      93   -0.3550
121  seasonal_naive52      93    0.0814
123       croston_sba      93   -0.0471
125               tsb      93   -0.3353
127            lgbm_w      93   -0.0351
129          all_zero      97   -1.0000
131             naive      97    0.1752
133             mean4      97    0.0124
135            mean13      97   -0.2208
137  seasonal_naive52      97    0.0024
139       croston_sba      97    0.0911
141               tsb      97   -0.2100
143            lgbm_w      97    0.0522
145          all_zero     101   -1.0000
147             naive     101   -0.0273
149             mean4     101   -0.1573
151            mean13     101   -0.2610
153  seasonal_naive52     101    0.0910
155       croston_sba     101   -0.0093
157               tsb     101   -0.2706
159            lgbm_w     101   -0.0429

== prob
             pinball  coverage_80
model                            
lgbm_w        3.1368       0.9151
empirical_q   3.5311       0.9098
all_zero      5.7710       0.7447
{
  "n_weeks": 105,
  "origins": [
    65,
    69,
    73,
    77,
    81,
    85,
    89,
    93,
    97,
    101
  ],
  "n_series": 9414,
  "zero_share_weekly": 0.7571489271298066,
  "sum_actuals": 4346261.0,
  "train_seconds": {
    "65": 26.63373637199402,
    "69": 28.916824102401733,
    "73": 32.712782859802246,
    "77": 31.555777311325073,
    "81": 40.24712681770325,
    "85": 38.05670499801636,
    "89": 45.51607871055603,
    "93": 47.928943157196045,
    "97": 61.07933235168457,
    "101": 60.90297603607178
  },
  "best_iteration": {
    "65": 111,
    "69": 167,
    "73": 159,
    "77": 58,
    "81": 226,
    "85": 71,
    "89": 149,
    "93": 167,
    "97": 400,
    "101": 237
  },
  "verdict": {
    "best_baseline_by_mase": "mean4",
    "origins_lgbm_beats_best_baseline_mase": "0/10",
    "c1_mase_5pct_and_7_of_10": false,
    "c2_wape_le_0.90": true,
    "c3_bias": true,
    "c4_pinball_and_coverage": false,
    "serving_candidate": false
  }
}
```
