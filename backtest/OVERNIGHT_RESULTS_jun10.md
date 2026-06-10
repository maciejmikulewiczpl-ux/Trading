# Overnight edge-hunt — Wed Jun 10 15:36:57 PDT 2026

## 1. stocks-in-play gates (RVOL/VWAP/barvol)

=== 730d: 487 eval sessions (of 502), OOS split 2025-06-16 ===
    pool: 2416 tight-OR trades with RVOL history (of 2487 tight)
    RVOL bucket           n  avg$/tr   win%
    0-0.5             1074     +2.6    42%
    0.5-1              830     +4.5    43%
    1-1.5              321     +3.2    41%
    1.5-2              112     +7.6    47%
    2+                  79     +7.1    48%

  -- slippage 1.0x --
  arm                trades      PnL$  Sharpe   maxDD$      h1 PnL   h2 PnL
  -------------------------------------------------------------------------
  base               2416    +8,914    2.43     -737      +6,637   +2,277
  rvol>=1.0           512    +2,439    1.70     -426      +1,618     +820
  rvol>=1.5           191    +1,411    1.79     -224        +838     +573
  rvol>=2.0            79      +562    1.20     -145        +407     +155
  vwap_confirm       2416    +8,914    2.43     -737      +6,637   +2,277
  barvol>=1.5         575    +1,953    1.39     -644      +2,015      -61
  rvol1.0+vwap        512    +2,439    1.70     -426      +1,618     +820

  -- slippage 1.5x --
  arm                trades      PnL$  Sharpe   maxDD$      h1 PnL   h2 PnL
  -------------------------------------------------------------------------
  base               2416    +6,396    1.75     -916      +5,144   +1,253
  rvol>=1.0           512    +2,013    1.41     -468      +1,380     +633
  rvol>=1.5           191    +1,259    1.61     -234        +750     +509
  rvol>=2.0            79      +505    1.08     -150        +373     +132
  vwap_confirm       2416    +6,396    1.75     -916      +5,144   +1,253
  barvol>=1.5         575    +1,300    0.93     -842      +1,591     -291
  rvol1.0+vwap        512    +2,013    1.41     -468      +1,380     +633

=== 180d: 108 eval sessions (of 123), OOS split 2026-03-19 ===
    pool: 207 tight-OR trades with RVOL history (of 300 tight)
    RVOL bucket           n  avg$/tr   win%
    0-0.5               82     -0.2    34%
    0.5-1               82     +8.6    45%
    1-1.5               35     +7.1    49%
    1.5-2                6     +6.4    50%
    2+                   2    +60.8    50%

  -- slippage 1.0x --
  arm                trades      PnL$  Sharpe   maxDD$      h1 PnL   h2 PnL
  -------------------------------------------------------------------------
  base                207    +1,096    2.13     -332        +266     +830
  rvol>=1.0            43      +408    2.27     -140         +10     +397
  rvol>=1.5             8      +160    1.43      -48         +50     +110
  rvol>=2.0             2      +122    1.30      -20         -20     +142
  vwap_confirm        207    +1,096    2.13     -332        +266     +830
  barvol>=1.5          43      +105    0.68     -274        -321     +426
  rvol1.0+vwap         43      +408    2.27     -140         +10     +397

  -- slippage 1.5x --
  arm                trades      PnL$  Sharpe   maxDD$      h1 PnL   h2 PnL
  -------------------------------------------------------------------------
  base                207      +906    1.77     -379        +129     +777
  rvol>=1.0            43      +378    2.12     -142         -10     +388
  rvol>=1.5             8      +156    1.41      -48         +48     +109
  rvol>=2.0             2      +120    1.29      -20         -20     +141
  vwap_confirm        207      +906    1.77     -379        +129     +777
  barvol>=1.5          43       +62    0.39     -300        -351     +413
  rvol1.0+vwap         43      +378    2.12     -142         -10     +388

Pre-registered gate: Sharpe >= base+0.10 AND maxDD <= base AND PnL >= 0.85x
base AND h2 >= base h2 - 10%, in BOTH windows at BOTH slips. Otherwise reject.

## 2. time-stop

=== 730d: 502 sessions, OOS split 2025-06-05 ===

  -- slippage 1.0x --
  arm              trades  scr%      PnL$  Sharpe   maxDD$      h1 PnL   h2 PnL
  -----------------------------------------------------------------------------
  trail (base)       2487    0%    +9,527    2.52     -737      +7,428   +2,099
  ts45m/+0.25R       2487    0%    +9,527    2.52     -737      +7,428   +2,099
  ts60m/+0.25R       2487    0%    +9,527    2.52     -737      +7,428   +2,099
  ts60m/+0.5R        2487    0%    +9,527    2.52     -737      +7,428   +2,099
  ts90m/+0.5R        2487    0%    +9,527    2.52     -737      +7,428   +2,099
  ts120m/+0.5R       2487    0%    +9,527    2.52     -737      +7,428   +2,099

  -- slippage 1.5x --
  arm              trades  scr%      PnL$  Sharpe   maxDD$      h1 PnL   h2 PnL
  -----------------------------------------------------------------------------
  trail (base)       2487    0%    +6,932    1.84     -916      +5,912   +1,020
  ts45m/+0.25R       2487    0%    +6,932    1.84     -916      +5,912   +1,020
  ts60m/+0.25R       2487    0%    +6,932    1.84     -916      +5,912   +1,020
  ts60m/+0.5R        2487    0%    +6,932    1.84     -916      +5,912   +1,020
  ts90m/+0.5R        2487    0%    +6,932    1.84     -916      +5,912   +1,020
  ts120m/+0.5R       2487    0%    +6,932    1.84     -916      +5,912   +1,020

=== 180d: 123 sessions, OOS split 2026-03-09 ===

  -- slippage 1.0x --
  arm              trades  scr%      PnL$  Sharpe   maxDD$      h1 PnL   h2 PnL
  -----------------------------------------------------------------------------
  trail (base)        300    0%    +1,060    1.60     -609        +296     +764
  ts45m/+0.25R        300    0%    +1,060    1.60     -609        +296     +764
  ts60m/+0.25R        300    0%    +1,060    1.60     -609        +296     +764
  ts60m/+0.5R         300    0%    +1,060    1.60     -609        +296     +764
  ts90m/+0.5R         300    0%    +1,060    1.60     -609        +296     +764
  ts120m/+0.5R        300    0%    +1,060    1.60     -609        +296     +764

  -- slippage 1.5x --
  arm              trades  scr%      PnL$  Sharpe   maxDD$      h1 PnL   h2 PnL
  -----------------------------------------------------------------------------
  trail (base)        300    0%      +780    1.19     -644         +88     +693
  ts45m/+0.25R        300    0%      +780    1.19     -644         +88     +693
  ts60m/+0.25R        300    0%      +780    1.19     -644         +88     +693
  ts60m/+0.5R         300    0%      +780    1.19     -644         +88     +693
  ts90m/+0.5R         300    0%      +780    1.19     -644         +88     +693
  ts120m/+0.5R        300    0%      +780    1.19     -644         +88     +693

Pre-registered gate: Sharpe AND PnL >= base, maxDD <= base, h2 not worse,
in BOTH windows at 1.0x; ordering must hold at 1.5x. Otherwise reject.

## 3. SPY intraday momentum

=== 730d ===

  SPY (499 sessions)  arm         n  win%  avg bps  PnL$@10k  Sharpe  maxDD$    h2 PnL
           fh        499   47%    -1.64      -820   -1.35    -987      -246
           r12       490   45%    -2.28    -1,119   -1.84  -1,116      -161
           fh+r12    255   49%    -1.94      -494   -1.24    -580       +24

  QQQ (499 sessions)  arm         n  win%  avg bps  PnL$@10k  Sharpe  maxDD$    h2 PnL
           fh        499   46%    -2.15    -1,072   -1.44  -1,258      -226
           r12       490   43%    -3.14    -1,537   -2.07  -1,740        +6
           fh+r12    261   46%    -3.22      -840   -1.65  -1,001      +120

  corr(daily $, tight-OR ORB daily $), 730d:
    SPY fh      : +0.01  (485 common days)
    SPY r12     : +0.05  (485 common days)
    SPY fh+r12  : +0.04  (485 common days)
    QQQ fh      : -0.01  (485 common days)
    QQQ r12     : +0.02  (485 common days)
    QQQ fh+r12  : +0.00  (485 common days)

=== 180d ===

  SPY (121 sessions)  arm         n  win%  avg bps  PnL$@10k  Sharpe  maxDD$    h2 PnL
           fh        121   42%    -1.42      -172   -1.57    -267       +61
           r12       120   47%    -1.15      -138   -1.27    -195      -119
           fh+r12     64   47%    -0.66       -42   -0.52    -119       +31

  QQQ (121 sessions)  arm         n  win%  avg bps  PnL$@10k  Sharpe  maxDD$    h2 PnL
           fh        121   50%    -0.71       -86   -0.62    -265       +28
           r12       120   54%    +0.76       +91    0.66    -162      +114
           fh+r12     68   57%    +1.58      +107    1.02     -86      +121

  corr(daily $, tight-OR ORB daily $), 180d:
    SPY fh      : -0.01  (107 common days)
    SPY r12     : -0.11  (107 common days)
    SPY fh+r12  : -0.09  (107 common days)
    QQQ fh      : -0.01  (107 common days)
    QQQ r12     : -0.11  (107 common days)
    QQQ fh+r12  : -0.09  (107 common days)

Pre-registered bar: net PnL > 0, Sharpe >= 1.0, |corr with ORB| <= 0.30,
in BOTH windows -> candidate 2nd engine (own runner + paper test required).

# finished Wed Jun 10 15:40:04 PDT 2026

## 2-FIXED. time-stop (after µs/ns deadline bugfix — positional deadline)

=== 730d: 502 sessions, OOS split 2025-06-05 ===

  -- slippage 1.0x --
  arm              trades  scr%      PnL$  Sharpe   maxDD$      h1 PnL   h2 PnL
  -----------------------------------------------------------------------------
  trail (base)       2487    0%    +9,527    2.52     -737      +7,428   +2,099
  ts45m/+0.25R       2487   13%    +7,062    1.97   -1,011      +5,558   +1,504
  ts60m/+0.25R       2487    9%    +8,224    2.20     -899      +6,413   +1,811
  ts60m/+0.5R        2487   17%    +7,713    2.13     -834      +5,988   +1,725
  ts90m/+0.5R        2487   11%    +8,548    2.30     -766      +6,625   +1,924
  ts120m/+0.5R       2487    8%    +8,808    2.35     -741      +7,016   +1,792

  -- slippage 1.5x --
  arm              trades  scr%      PnL$  Sharpe   maxDD$      h1 PnL   h2 PnL
  -----------------------------------------------------------------------------
  trail (base)       2487    0%    +6,932    1.84     -916      +5,912   +1,020
  ts45m/+0.25R       2487   13%    +4,467    1.25   -1,438      +4,042     +425
  ts60m/+0.25R       2487    9%    +5,629    1.52   -1,329      +4,897     +732
  ts60m/+0.5R        2487   17%    +5,117    1.42   -1,189      +4,471     +646
  ts90m/+0.5R        2487   11%    +5,953    1.61   -1,167      +5,108     +845
  ts120m/+0.5R       2487    8%    +6,212    1.67   -1,101      +5,499     +713

=== 180d: 123 sessions, OOS split 2026-03-09 ===

  -- slippage 1.0x --
  arm              trades  scr%      PnL$  Sharpe   maxDD$      h1 PnL   h2 PnL
  -----------------------------------------------------------------------------
  trail (base)        300    0%    +1,060    1.60     -609        +296     +764
  ts45m/+0.25R        300   13%      +746    1.24     -591         +52     +694
  ts60m/+0.25R        300   10%      +944    1.43     -613        +228     +716
  ts60m/+0.5R         300   18%      +887    1.40     -587        +333     +554
  ts90m/+0.5R         300   11%      +880    1.34     -592        +257     +623
  ts120m/+0.5R        300    6%      +971    1.48     -629        +287     +684

  -- slippage 1.5x --
  arm              trades  scr%      PnL$  Sharpe   maxDD$      h1 PnL   h2 PnL
  -----------------------------------------------------------------------------
  trail (base)        300    0%      +780    1.19     -644         +88     +693
  ts45m/+0.25R        300   13%      +466    0.78     -630        -156     +622
  ts60m/+0.25R        300   10%      +664    1.02     -648         +20     +644
  ts60m/+0.5R         300   18%      +606    0.97     -622        +125     +482
  ts90m/+0.5R         300   11%      +600    0.93     -627         +49     +551
  ts120m/+0.5R        300    6%      +691    1.07     -664         +79     +612

Pre-registered gate: Sharpe AND PnL >= base, maxDD <= base, h2 not worse,
in BOTH windows at 1.0x; ordering must hold at 1.5x. Otherwise reject.

# rerun finished Wed Jun 10 15:44:16 PDT 2026

## 4. retest entry + same-direction re-entry (practitioner round)

=== 730d: 502 sessions, OOS split 2025-06-05  (2487 tight-OR trades) ===

  -- slippage 1.0x --
  arm              trades  fill%      PnL$  Sharpe   maxDD$      h1 PnL   h2 PnL
  ------------------------------------------------------------------------------
  base               2469   100%    +9,527    2.52     -737      +7,428   +2,099
  retest_ORhigh      1860    75%    +4,234    1.78     -622      +3,347     +887
  reenter_1x         2559   100%    +9,954    2.59     -807      +7,807   +2,147

  -- slippage 1.5x --
  arm              trades  fill%      PnL$  Sharpe   maxDD$      h1 PnL   h2 PnL
  ------------------------------------------------------------------------------
  base               2469   100%    +6,932    1.84     -916      +5,912   +1,020
  retest_ORhigh      1860    75%    +2,112    0.89   -1,311      +2,114       -3
  reenter_1x         2559   100%    +7,290    1.91     -955      +6,247   +1,043

=== 180d: 123 sessions, OOS split 2026-03-09  (300 tight-OR trades) ===

  -- slippage 1.0x --
  arm              trades  fill%      PnL$  Sharpe   maxDD$      h1 PnL   h2 PnL
  ------------------------------------------------------------------------------
  base                299   100%    +1,060    1.60     -609        +296     +764
  retest_ORhigh       251    84%      +836    1.55     -415        +187     +648
  reenter_1x          311   100%    +1,088    1.63     -620        +363     +725

  -- slippage 1.5x --
  arm              trades  fill%      PnL$  Sharpe   maxDD$      h1 PnL   h2 PnL
  ------------------------------------------------------------------------------
  base                299   100%      +780    1.19     -644         +88     +693
  retest_ORhigh       251    84%      +594    1.11     -449          +6     +588
  reenter_1x          311   100%      +798    1.21     -657        +149     +649

Gates: retest needs Sharpe AND PnL >= base (both windows, both slips).
reenter needs PnL >= base, Sharpe >= base-0.05, maxDD <= 1.15x base.

# round-2 finished Wed Jun 10 16:05:40 PDT 2026
