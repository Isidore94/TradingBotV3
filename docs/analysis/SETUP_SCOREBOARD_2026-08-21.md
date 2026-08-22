# Setup scoreboard — 2026-07-24 … 2026-08-21

**Read-only. This report promotes and demotes nothing.** plan.md Section 7
gate 2 requires an evidence window frozen *before* inspection; this window was
chosen after, so nothing measured here can move a rung. Its one forward-looking
output is the declared window in the last section.

Generated 2026-08-22T10:18:58-07:00. Zones: `trade_date` is a bare market date; `logged_at` is tz-aware −07:00 (desk local, America/Los_Angeles); `entry_time` is naive desk-local PT, and market time is PT + 3h (America/New_York).

## 1. Coverage, and what was excluded before anything was ranked
| stage | rows |
|---|---|
| scanned (all milestone rows) | 239,422 |
| `event_type == final` | 14,452 |
| in window | 6,907 |
| distinct sessions | 20 |
| excluded — no EOD close obtained | 1,164 |
| …of those, never advanced a bar | 251 |
| excluded — stop under 0.1% of entry | 212 |
| **usable** | **5,608** |

### 1a. The `close_r == 0` mass is a defect, not a population of scratches
1,164 of 6,907 in-window finals (16.9%) carry `close_r` exactly 0. **Every one of them has `eod_close` exactly equal to `entry_price`, and none of the settled finals does.** A real close does not land on the entry to the cent 1,164 times and never otherwise — the outcome writer defaults `eod_close` to the entry when it cannot read one.

251 of those never advanced a bar at all (`bars_elapsed == 0`); the rest have real excursions and were simply never closed out. Treating any of them as a scratch biases every mean **upward**, because the stopped-out ones — which should score about −1R — score 0 instead. They are excluded from every number below and counted here instead.

This is the single largest data-quality finding in the rebuild and it is an argument for fixing the writer, not for reading around it.

## 2. Intraday families (`intraday_bounce_outcomes.csv` finals)
Cells with n < 30 are listed but marked `reportable = False` and are not ranked. Every R appears as mean, 10% trimmed mean and median, with the stop-out rate beside it — a plain mean on a ratio with an unbounded numerator is exactly the statistic that produced the review's phantom −1.82R.

| cell | n | mean_r | trimmed_mean_r | median_r | stop_out_rate | p10_r | p90_r | reportable |
|---|---|---|---|---|---|---|---|---|
| regime_pause_rw | 195 | 0.354 | 0.417 | 0.45 | 71.3 | -1.837 | 2.279 | True |
| eod_vwap-impulse_retest_vwap_eod | 72 | 0.092 | 0.126 | 0.33 | 58.3 | -2.254 | 2.536 | True |
| regime_pause_rs | 266 | 0.237 | 0.123 | 0.016 | 60.9 | -2.997 | 3.495 | True |
| eod_vwap_upper_band | 36 | 0.04 | 0.11 | 0.141 | 44.4 | -1.832 | 2.146 | True |
| 10_candle_low | 128 | 0.116 | 0.084 | 0.056 | 43.0 | -2.071 | 2.379 | True |
| 10_candle_high | 76 | -0.082 | -0.001 | -0.0 | 51.3 | -1.822 | 1.481 | True |
| h1_green_to_yellow | 341 | -0.007 | -0.016 | -0.045 | 51.3 | -1.25 | 1.178 | True |
| h1_ema_15 | 197 | 0.079 | -0.028 | -0.009 | 37.6 | -1.292 | 1.585 | True |
| h1_sma_20 | 59 | 0.024 | -0.033 | -0.108 | 23.7 | -0.772 | 0.764 | True |
| eod_vwap-impulse_retest_vwap_eod-vwap | 122 | -0.138 | -0.058 | 0.038 | 51.6 | -2.308 | 1.954 | True |
| h1_ema10_bounce | 2739 | -0.121 | -0.084 | -0.091 | 78.2 | -2.182 | 1.91 | True |
| h1_blue_after_red | 1106 | -0.139 | -0.126 | -0.133 | 49.8 | -1.291 | 1.054 | True |
| dynamic_vwap_upper_band | 30 | -0.519 | -0.508 | -0.329 | 56.7 | -2.932 | 1.799 | True |
| vwap_upper_band | 32 | -0.538 | -0.569 | -0.663 | 62.5 | -1.802 | 0.561 | True |
| 10_candle_low-ema_21 | 2 | 6.01 | 6.01 | 6.01 | 0.0 | 3.012 | 9.007 | False |
| ema_21-impulse_retest_vwap_eod-vwap | 1 | 4.059 | 4.059 | 4.059 | 0.0 | 4.059 | 4.059 | False |
| ema_8 | 1 | 3.55 | 3.55 | 3.55 | 100.0 | 3.55 | 3.55 | False |
| 10_candle_high-eod_vwap_lower_band | 2 | 1.537 | 1.537 | 1.537 | 50.0 | 0.942 | 2.133 | False |
| prev_day_high | 2 | 1.5 | 1.5 | 1.5 | 50.0 | 1.486 | 1.514 | False |
| eod_vwap-eod_vwap_lower_band-impulse_retest_vwap_eod-vwap | 1 | 1.438 | 1.438 | 1.438 | 0.0 | 1.438 | 1.438 | False |
| ema8_grind_hod | 1 | 1.227 | 1.227 | 1.227 | 100.0 | 1.227 | 1.227 | False |
| eod_vwap-eod_vwap_upper_band-impulse_retest_vwap_eod-vwap | 3 | 1.132 | 1.132 | 1.596 | 0.0 | -0.057 | 2.136 | False |
| 10_candle_high-dynamic_vwap_lower_band | 6 | 0.872 | 0.872 | 0.713 | 16.7 | 0.391 | 1.511 | False |
| vwap_lower_band | 12 | 0.641 | 0.687 | 0.581 | 16.7 | -0.607 | 1.717 | False |
| dynamic_vwap_upper_band-eod_vwap_upper_band | 2 | 0.576 | 0.576 | 0.576 | 0.0 | -0.126 | 1.279 | False |
| ema_21-vwap_upper_band | 1 | 0.507 | 0.507 | 0.507 | 0.0 | 0.507 | 0.507 | False |
| dynamic_vwap_upper_band-eod_vwap-impulse_retest_vwap_eod | 1 | 0.497 | 0.497 | 0.497 | 0.0 | 0.497 | 0.497 | False |
| eod_vwap_lower_band | 14 | 0.523 | 0.448 | 0.435 | 50.0 | -0.801 | 2.078 | False |
| dynamic_vwap_lower_band-eod_vwap_lower_band-vwap_lower_band | 1 | 0.43 | 0.43 | 0.43 | 0.0 | 0.43 | 0.43 | False |
| impulse_retest_vwap_eod-vwap | 11 | 0.25 | 0.398 | 1.079 | 72.7 | -2.95 | 2.57 | False |
| 10_candle_low-dynamic_vwap_upper_band | 6 | 0.363 | 0.363 | 0.965 | 33.3 | -1.429 | 1.552 | False |
| dynamic_vwap_lower_band-prev_day_low | 2 | 0.304 | 0.304 | 0.304 | 0.0 | 0.067 | 0.541 | False |
| eod_vwap_upper_band-vwap_upper_band | 8 | 0.289 | 0.289 | 0.413 | 37.5 | -1.786 | 2.053 | False |
| ema_21 | 8 | 0.252 | 0.252 | -0.026 | 50.0 | -0.396 | 1.128 | False |
| dynamic_vwap_lower_band | 22 | 0.274 | 0.198 | 0.109 | 40.9 | -0.916 | 1.711 | False |
| 10_candle_low-eod_vwap_upper_band | 2 | 0.142 | 0.142 | 0.142 | 100.0 | -1.517 | 1.801 | False |
| ema_15-vwap_upper_band | 1 | 0.079 | 0.079 | 0.079 | 0.0 | 0.079 | 0.079 | False |
| h1_ema_15-h1_sma_20 | 25 | -0.005 | 0.018 | -0.023 | 12.0 | -0.337 | 0.298 | False |
| eod_vwap_lower_band-vwap_lower_band | 5 | -0.153 | -0.153 | 0.011 | 40.0 | -1.144 | 0.774 | False |
| h1_ema_15-impulse_retest_vwap_eod-vwap | 1 | -0.197 | -0.197 | -0.197 | 0.0 | -0.197 | -0.197 | False |
| orb_breakout | 24 | -0.195 | -0.2 | -0.386 | 58.3 | -1.09 | 1.275 | False |
| dynamic_vwap_upper_band-eod_vwap-impulse_retest_vwap_eod-vwap | 3 | -0.216 | -0.216 | -0.333 | 0.0 | -0.548 | 0.162 | False |
| 10_candle_low-dynamic_vwap_upper_band-prev_day_high | 1 | -0.242 | -0.242 | -0.242 | 0.0 | -0.242 | -0.242 | False |
| 10_candle_low-dynamic_vwap_upper_band-eod_vwap_upper_band-vwap_upper_band | 1 | -0.277 | -0.277 | -0.277 | 0.0 | -0.277 | -0.277 | False |
| prev_day_low | 3 | -0.383 | -0.383 | -1.401 | 66.7 | -1.401 | 1.043 | False |
| 10_candle_low-eod_vwap-impulse_retest_vwap_eod-vwap | 1 | -0.411 | -0.411 | -0.411 | 0.0 | -0.411 | -0.411 | False |
| ema_15 | 11 | -0.609 | -0.616 | -0.554 | 63.6 | -2.067 | 0.616 | False |
| orb_breakdown | 16 | -0.719 | -0.78 | -0.916 | 56.2 | -1.884 | 0.176 | False |
| h1_ema_15-vwap_upper_band | 1 | -1.158 | -1.158 | -1.158 | 100.0 | -1.158 | -1.158 | False |
| dynamic_vwap_lower_band-eod_vwap-impulse_retest_vwap_eod-vwap | 1 | -1.397 | -1.397 | -1.397 | 100.0 | -1.397 | -1.397 | False |
| 10_candle_low-prev_day_high | 2 | -1.733 | -1.733 | -1.733 | 100.0 | -2.434 | -1.032 | False |
| 10_candle_low-eod_vwap_upper_band-vwap_upper_band | 2 | -1.862 | -1.862 | -1.862 | 50.0 | -3.064 | -0.66 | False |
| 10_candle_low-dynamic_vwap_upper_band-eod_vwap_upper_band | 1 | -3.222 | -3.222 | -3.222 | 100.0 | -3.222 | -3.222 | False |
| dynamic_vwap_lower_band-eod_vwap-impulse_retest_vwap_eod | 1 | -3.393 | -3.393 | -3.393 | 100.0 | -3.393 | -3.393 | False |

### 2a. By market environment — the axis the review called starved
`market_environment` is present on 100% of these rows, from `context_json`. The review reported this axis at n=130 because it read the review store; the outcome store carries it on every row.

| cell | n | mean_r | trimmed_mean_r | median_r | stop_out_rate | p10_r | p90_r | reportable |
|---|---|---|---|---|---|---|---|---|
| bullish_strong | 1139 | 0.106 | 0.085 | 0.053 | 60.3 | -1.767 | 2.027 | True |
| bullish_weak | 1643 | -0.025 | -0.031 | -0.056 | 60.7 | -1.583 | 1.536 | True |
| bearish_strong | 1119 | -0.097 | -0.06 | -0.021 | 70.6 | -2.116 | 2.083 | True |
| neutral_chop | 1049 | -0.14 | -0.142 | -0.143 | 64.7 | -2.0 | 1.706 | True |
| bearish_weak | 658 | -0.257 | -0.166 | -0.163 | 67.3 | -2.013 | 1.613 | True |

### 2b. By session RVOL bucket
| cell | n | mean_r | trimmed_mean_r | median_r | stop_out_rate | p10_r | p90_r | reportable |
|---|---|---|---|---|---|---|---|---|
| 0.8-1.2 | 330 | 0.175 | 0.188 | 0.184 | 50.9 | -1.518 | 2.019 | True |
| 1.2-2.0 | 253 | 0.101 | 0.053 | -0.041 | 60.5 | -1.847 | 2.105 | True |
| <0.8 | 1245 | -0.003 | -0.026 | -0.041 | 50.4 | -1.679 | 1.591 | True |
| >2.0 | 150 | 0.081 | -0.042 | -0.171 | 64.0 | -2.163 | 2.784 | True |

### 2c. By sector
| cell | n | mean_r | trimmed_mean_r | median_r | stop_out_rate | p10_r | p90_r | reportable |
|---|---|---|---|---|---|---|---|---|
| (unknown) | 30 | 0.559 | 0.457 | 0.458 | 73.3 | -1.515 | 2.413 | True |
| Communication Services | 243 | 0.097 | 0.085 | 0.08 | 62.6 | -1.711 | 2.132 | True |
| Healthcare | 1076 | 0.113 | 0.06 | -0.021 | 68.6 | -1.694 | 2.389 | True |
| Financial Services | 910 | 0.027 | 0.052 | 0.065 | 61.8 | -1.682 | 1.571 | True |
| Consumer Cyclical | 579 | 0.015 | -0.045 | -0.086 | 62.2 | -1.796 | 1.952 | True |
| Energy | 390 | -0.176 | -0.063 | -0.04 | 67.4 | -2.168 | 1.864 | True |
| Consumer Defensive | 175 | -0.047 | -0.087 | -0.108 | 62.3 | -1.602 | 1.819 | True |
| Technology | 1023 | -0.134 | -0.109 | -0.12 | 61.3 | -2.103 | 1.781 | True |
| Industrials | 560 | -0.132 | -0.11 | -0.108 | 63.4 | -1.811 | 1.674 | True |
| Real Estate | 301 | -0.33 | -0.245 | -0.267 | 64.1 | -2.158 | 1.431 | True |
| Basic Materials | 210 | -0.457 | -0.27 | -0.205 | 65.2 | -2.177 | 1.326 | True |
| Utilities | 111 | -0.522 | -0.408 | -0.5 | 70.3 | -2.0 | 1.038 | True |

## 3. Swing families vs their own control (`setup_playbook_episodes.csv`)
Control is `baseline_every5`, the file's own equal-weight every-fifth-session entry over the same names and window. Only the difference from it is attributable to the setup.

| cell | n | mean_r | trimmed_mean_r | median_r | baseline_trimmed_r | lift_vs_baseline | reportable |
|---|---|---|---|---|---|---|---|
| volume_thrust / LONG | 49 | -0.004 | -0.099 | -0.146 | -0.573 | 0.474 | True |
| inside_day_break / LONG | 457 | 0.208 | -0.2 | -0.457 | -0.573 | 0.373 | True |
| ema21_pullback_uptrend / LONG | 853 | 0.043 | -0.239 | -0.782 | -0.573 | 0.334 | True |
| vwap_reclaim / LONG | 1099 | 0.096 | -0.255 | -0.72 | -0.573 | 0.318 | True |
| gap_up_hold / LONG | 794 | 0.163 | -0.288 | -1.01 | -0.573 | 0.285 | True |
| sma50_reclaim / LONG | 461 | 0.164 | -0.299 | -0.679 | -0.573 | 0.274 | True |
| first_dev_breakout / LONG | 1167 | 0.066 | -0.333 | -1.007 | -0.573 | 0.24 | True |
| quiet_pullback_resume / LONG | 325 | 0.131 | -0.333 | -1.012 | -0.573 | 0.24 | True |
| post_earnings_gap_hold3 / LONG | 78 | 0.045 | -0.346 | -0.914 | -0.573 | 0.227 | True |
| range5d_break_above_vwap / LONG | 1289 | 0.01 | -0.36 | -1.007 | -0.573 | 0.213 | True |
| second_dev_breakout / LONG | 931 | -0.013 | -0.385 | -0.745 | -0.573 | 0.188 | True |
| sma50_reclaim / SHORT | 446 | -0.242 | -0.418 | -0.778 | -0.573 | 0.155 | True |
| vwap_bounce / LONG | 900 | 0.243 | -0.426 | -1.02 | -0.573 | 0.147 | True |
| sma200_reclaim / LONG | 284 | -0.202 | -0.428 | -1.011 | -0.573 | 0.145 | True |
| power_trend_pullback / LONG | 442 | -0.059 | -0.446 | -1.021 | -0.573 | 0.127 | True |
| ema8_pullback_uptrend / LONG | 772 | -0.086 | -0.449 | -1.019 | -0.573 | 0.124 | True |
| band_test_rebound / LONG | 596 | -0.106 | -0.469 | -1.014 | -0.573 | 0.104 | True |
| ema15_bounce_trend / LONG | 832 | -0.029 | -0.469 | -1.022 | -0.573 | 0.104 | True |
| high_252_breakout / LONG | 243 | -0.298 | -0.471 | -0.68 | -0.573 | 0.102 | True |
| post_earnings_volume_break / SHORT | 35 | -0.408 | -0.477 | -1.007 | -0.573 | 0.096 | True |
| range5d_break_above_vwap / SHORT | 809 | -0.312 | -0.496 | -1.008 | -0.573 | 0.077 | True |
| volume_thrust / SHORT | 37 | -0.506 | -0.526 | -1.011 | -0.573 | 0.047 | True |
| band_test_rebound / SHORT | 296 | -0.275 | -0.532 | -1.015 | -0.573 | 0.041 | True |
| golden_pullback_sma50 / LONG | 209 | -0.238 | -0.532 | -1.022 | -0.573 | 0.041 | True |
| vwap_reclaim / SHORT | 893 | -0.271 | -0.533 | -1.013 | -0.573 | 0.04 | True |
| first_dev_bounce / LONG | 879 | 0.072 | -0.539 | -1.025 | -0.573 | 0.034 | True |
| inside_day_break / SHORT | 369 | -0.347 | -0.539 | -1.013 | -0.573 | 0.034 | True |
| quiet_pullback_resume / SHORT | 201 | -0.356 | -0.555 | -1.012 | -0.573 | 0.018 | True |
| sma200_reclaim / SHORT | 222 | -0.407 | -0.562 | -1.008 | -0.573 | 0.011 | True |
| second_dev_power_hold / LONG | 280 | -0.097 | -0.599 | -1.026 | -0.573 | -0.026 | True |
| first_dev_breakout / SHORT | 828 | -0.428 | -0.63 | -1.015 | -0.573 | -0.057 | True |
| breakout_retest_252 / LONG | 157 | -0.405 | -0.636 | -1.021 | -0.573 | -0.063 | True |
| ema21_pullback_uptrend / SHORT | 461 | -0.405 | -0.64 | -1.021 | -0.573 | -0.067 | True |
| second_dev_breakout / SHORT | 614 | -0.454 | -0.644 | -1.015 | -0.573 | -0.071 | True |
| gap_up_hold / SHORT | 456 | -0.486 | -0.655 | -1.013 | -0.573 | -0.082 | True |
| high_252_breakout / SHORT | 60 | -0.461 | -0.668 | -1.011 | -0.573 | -0.095 | True |
| golden_pullback_sma50 / SHORT | 151 | -0.467 | -0.68 | -1.028 | -0.573 | -0.107 | True |
| vwap_bounce / SHORT | 750 | -0.373 | -0.714 | -1.036 | -0.573 | -0.141 | True |
| ema8_pullback_uptrend / SHORT | 502 | -0.471 | -0.715 | -1.029 | -0.573 | -0.142 | True |
| power_trend_pullback / SHORT | 277 | -0.548 | -0.748 | -1.023 | -0.573 | -0.175 | True |
| breakout_retest_252 / SHORT | 32 | -0.551 | -0.756 | -1.031 | -0.573 | -0.183 | True |
| second_dev_power_hold / SHORT | 132 | -0.539 | -0.769 | -1.031 | -0.573 | -0.196 | True |
| post_earnings_gap_hold3 / SHORT | 79 | -0.675 | -0.789 | -1.037 | -0.573 | -0.216 | True |
| first_dev_bounce / SHORT | 626 | -0.48 | -0.79 | -1.036 | -0.573 | -0.217 | True |
| ema15_bounce_trend / SHORT | 448 | -0.572 | -0.796 | -1.031 | -0.573 | -0.223 | True |
| weekly_weak_volume_reclaim / LONG | 11 | 0.36 | 0.314 | 0.676 | -0.573 | 0.887 | False |
| post_earnings_volume_break / LONG | 29 | 0.04 | -0.124 | -0.103 | -0.573 | 0.449 | False |
| weekly_weak_volume_reclaim / SHORT | 7 | -0.174 | -0.174 | -0.036 | -0.573 | 0.399 | False |
| golden_pullback_sma50_vol / LONG | 3 | -0.187 | -0.187 | -1.006 | -0.573 | 0.386 | False |
| golden_pullback_sma50_vol / SHORT | 5 | -0.361 | -0.361 | -1.014 | -0.573 | 0.212 | False |
| post_earnings_avwape_first_tag / LONG | 2 | -1.361 | -1.361 | -1.361 | -0.573 | -0.788 | False |

**Read this table as relative, never as absolute.** The control's own trimmed R is -0.573, so a positive `lift_vs_baseline` means *lost less than the control*, not *made money*. Two features of the block deserve stating before any of it is quoted: the median `net_r` sits at roughly −1.0 across most families, which means more than half of every family's episodes are full stop-outs; and the plain mean sits far above the trimmed mean nearly everywhere, which means what positive numbers exist are carried by a thin tail of large winners. A family here is a candidate for measurement, not a candidate for size.

## 4. What this report does NOT establish
- **No promotion or demotion.** Gate 2 is unsatisfiable post-hoc.
- **No causal claim.** The environment and RVOL splits are conditional descriptions of one window, not evidence that a family works *because* of a regime.
- **Nothing from a cell under n=30.** Those rows are printed so the thinness is visible, not so they can be read.
- **Nothing about the excluded rows.** The unsettled mass is a writer defect with an unknown outcome, and unknown is not zero.

## 5. The declared window for the next inspection
Everything above is post-hoc. This is the part that is not: the window below is frozen **now**, before it is measured, which is the only route by which any number in this file ever becomes plan.md Section 7 gate-2 eligible.

- **declared_on**: 2026-08-22
- **starts**: the first session after this report is committed
- **length_sessions**: 40
- **must_span**: bullish, bearish, chop
- **primary_metric**: trimmed-mean net R per (family, side), cells n>=30
- **control**: baseline_every5
- **exclusions_fixed_in_advance**: risk_per_share < 0.1% of entry, close_r == 0 with eod_close == entry_price (no EOD close obtained)
- **decision_rule**: no promotion or demotion from this report or the next one alone; the declared window produces the first gate-2-eligible evidence and the trader decides what it is evidence FOR
