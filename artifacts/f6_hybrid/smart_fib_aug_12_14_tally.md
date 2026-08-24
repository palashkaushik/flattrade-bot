# Smart Fib Historical Trade Tally

- Cache: `artifacts\flattrade_day_cache_smart_fib`
- Dates: 2026-08-12, 2026-08-13, 2026-08-14
- Entry/exit matching: every reported trade must have both timestamps in its downloaded option contract rows.

## Download Coverage

| Date | Spot rows | Contracts | Option rows |
|---|---:|---:|---:|
| 2026-08-12 | 376 | 14 | 10500 |
| 2026-08-13 | 376 | 10 | 7500 |
| 2026-08-14 | 376 | 10 | 7500 |

## Aggregate Tally

| Configuration | Trades | Matched | Net points | Net Rs |
|---|---:|---:|---:|---:|
| `smart-fib|1m|tp0.29|sl1.155` | 12 | 12 | +41.55 | +2575.75 |
| `smart-fib|1m|tp0.29|sl1.25` | 11 | 11 | +27.00 | +1639.78 |
| `smart-fib|2m|tp0.29|sl1.155` | 12 | 12 | +41.55 | +2575.75 |
| `smart-fib|2m|tp0.29|sl1.25` | 11 | 11 | +27.00 | +1639.78 |
| `smart-fib|3m|tp0.29|sl1.155` | 12 | 12 | +41.55 | +2575.75 |
| `smart-fib|3m|tp0.29|sl1.25` | 11 | 11 | +27.00 | +1639.78 |
| `smart-fib|5m|tp0.29|sl1.155` | 12 | 12 | +41.55 | +2575.75 |
| `smart-fib|5m|tp0.29|sl1.25` | 11 | 11 | +27.00 | +1639.78 |
| `smart-fib|combined|tp0.29|sl1.155` | 12 | 12 | +41.55 | +2575.75 |
| `smart-fib|combined|tp0.29|sl1.25` | 11 | 11 | +27.00 | +1639.78 |

## Daily Details

### `smart-fib|1m|tp0.29|sl1.155`

- 2026-08-12: 2 trades, 2 matched, net +10.90 points / Rs +685.68
  - 10:34 -> 10:48 PE `NIFTY18AUG26P24400` TP +18.60 points / Rs +1196.82
  - 12:59 -> 13:19 PE `NIFTY18AUG26P24300` SL -7.70 points / Rs -511.14
- 2026-08-13: 3 trades, 3 matched, net +6.95 points / Rs +419.24
  - 10:49 -> 10:52 PE `NIFTY18AUG26P24400` SL -3.00 points / Rs -205.88
  - 11:28 -> 11:36 PE `NIFTY18AUG26P24350` SL -7.55 points / Rs -499.03
  - 14:27 -> 14:56 PE `NIFTY18AUG26P24500` TP +17.50 points / Rs +1124.15
- 2026-08-14: 7 trades, 7 matched, net +23.70 points / Rs +1470.83
  - 10:23 -> 10:31 CE `NIFTY18AUG26C24350` SL -5.90 points / Rs -392.67
  - 10:51 -> 10:59 PE `NIFTY18AUG26P24350` TP +14.20 points / Rs +914.57
  - 11:20 -> 12:07 PE `NIFTY18AUG26P24350` SL -0.35 points / Rs -30.56
  - 12:19 -> 12:21 CE `NIFTY18AUG26C24300` TP -0.75 points / Rs -61.49
  - 12:44 -> 13:42 CE `NIFTY18AUG26C24300` TP +16.70 points / Rs +1072.72
  - 14:12 -> 14:18 CE `NIFTY18AUG26C24350` SL -3.10 points / Rs -213.69
  - 14:55 -> 15:00 PE `NIFTY18AUG26P24400` TP +2.90 points / Rs +181.95

### `smart-fib|1m|tp0.29|sl1.25`

- 2026-08-12: 2 trades, 2 matched, net -0.40 points / Rs -48.06
  - 10:34 -> 10:48 PE `NIFTY18AUG26P24400` TP +18.60 points / Rs +1196.82
  - 12:59 -> 14:30 PE `NIFTY18AUG26P24300` SL -19.00 points / Rs -1244.88
- 2026-08-13: 2 trades, 2 matched, net +12.50 points / Rs +788.41
  - 10:49 -> 11:35 PE `NIFTY18AUG26P24400` SL -5.00 points / Rs -335.74
  - 14:27 -> 14:56 PE `NIFTY18AUG26P24500` TP +17.50 points / Rs +1124.15
- 2026-08-14: 7 trades, 7 matched, net +14.90 points / Rs +899.43
  - 10:23 -> 10:32 CE `NIFTY18AUG26C24350` SL -9.45 points / Rs -623.18
  - 10:51 -> 10:59 PE `NIFTY18AUG26P24350` TP +14.20 points / Rs +914.57
  - 11:20 -> 12:11 PE `NIFTY18AUG26P24350` SL -3.05 points / Rs -205.87
  - 12:19 -> 12:21 CE `NIFTY18AUG26C24300` TP -0.75 points / Rs -61.49
  - 12:44 -> 13:42 CE `NIFTY18AUG26C24300` TP +16.70 points / Rs +1072.72
  - 14:12 -> 14:21 CE `NIFTY18AUG26C24350` SL -5.65 points / Rs -379.27
  - 14:55 -> 15:00 PE `NIFTY18AUG26P24400` TP +2.90 points / Rs +181.95

### `smart-fib|2m|tp0.29|sl1.155`

- 2026-08-12: 2 trades, 2 matched, net +10.90 points / Rs +685.68
  - 10:34 -> 10:48 PE `NIFTY18AUG26P24400` TP +18.60 points / Rs +1196.82
  - 12:59 -> 13:19 PE `NIFTY18AUG26P24300` SL -7.70 points / Rs -511.14
- 2026-08-13: 3 trades, 3 matched, net +6.95 points / Rs +419.24
  - 10:49 -> 10:52 PE `NIFTY18AUG26P24400` SL -3.00 points / Rs -205.88
  - 11:28 -> 11:36 PE `NIFTY18AUG26P24350` SL -7.55 points / Rs -499.03
  - 14:27 -> 14:56 PE `NIFTY18AUG26P24500` TP +17.50 points / Rs +1124.15
- 2026-08-14: 7 trades, 7 matched, net +23.70 points / Rs +1470.83
  - 10:23 -> 10:31 CE `NIFTY18AUG26C24350` SL -5.90 points / Rs -392.67
  - 10:51 -> 10:59 PE `NIFTY18AUG26P24350` TP +14.20 points / Rs +914.57
  - 11:20 -> 12:07 PE `NIFTY18AUG26P24350` SL -0.35 points / Rs -30.56
  - 12:19 -> 12:21 CE `NIFTY18AUG26C24300` TP -0.75 points / Rs -61.49
  - 12:44 -> 13:42 CE `NIFTY18AUG26C24300` TP +16.70 points / Rs +1072.72
  - 14:12 -> 14:18 CE `NIFTY18AUG26C24350` SL -3.10 points / Rs -213.69
  - 14:55 -> 15:00 PE `NIFTY18AUG26P24400` TP +2.90 points / Rs +181.95

### `smart-fib|2m|tp0.29|sl1.25`

- 2026-08-12: 2 trades, 2 matched, net -0.40 points / Rs -48.06
  - 10:34 -> 10:48 PE `NIFTY18AUG26P24400` TP +18.60 points / Rs +1196.82
  - 12:59 -> 14:30 PE `NIFTY18AUG26P24300` SL -19.00 points / Rs -1244.88
- 2026-08-13: 2 trades, 2 matched, net +12.50 points / Rs +788.41
  - 10:49 -> 11:35 PE `NIFTY18AUG26P24400` SL -5.00 points / Rs -335.74
  - 14:27 -> 14:56 PE `NIFTY18AUG26P24500` TP +17.50 points / Rs +1124.15
- 2026-08-14: 7 trades, 7 matched, net +14.90 points / Rs +899.43
  - 10:23 -> 10:32 CE `NIFTY18AUG26C24350` SL -9.45 points / Rs -623.18
  - 10:51 -> 10:59 PE `NIFTY18AUG26P24350` TP +14.20 points / Rs +914.57
  - 11:20 -> 12:11 PE `NIFTY18AUG26P24350` SL -3.05 points / Rs -205.87
  - 12:19 -> 12:21 CE `NIFTY18AUG26C24300` TP -0.75 points / Rs -61.49
  - 12:44 -> 13:42 CE `NIFTY18AUG26C24300` TP +16.70 points / Rs +1072.72
  - 14:12 -> 14:21 CE `NIFTY18AUG26C24350` SL -5.65 points / Rs -379.27
  - 14:55 -> 15:00 PE `NIFTY18AUG26P24400` TP +2.90 points / Rs +181.95

### `smart-fib|3m|tp0.29|sl1.155`

- 2026-08-12: 2 trades, 2 matched, net +10.90 points / Rs +685.68
  - 10:34 -> 10:48 PE `NIFTY18AUG26P24400` TP +18.60 points / Rs +1196.82
  - 12:59 -> 13:19 PE `NIFTY18AUG26P24300` SL -7.70 points / Rs -511.14
- 2026-08-13: 3 trades, 3 matched, net +6.95 points / Rs +419.24
  - 10:49 -> 10:52 PE `NIFTY18AUG26P24400` SL -3.00 points / Rs -205.88
  - 11:28 -> 11:36 PE `NIFTY18AUG26P24350` SL -7.55 points / Rs -499.03
  - 14:27 -> 14:56 PE `NIFTY18AUG26P24500` TP +17.50 points / Rs +1124.15
- 2026-08-14: 7 trades, 7 matched, net +23.70 points / Rs +1470.83
  - 10:23 -> 10:31 CE `NIFTY18AUG26C24350` SL -5.90 points / Rs -392.67
  - 10:51 -> 10:59 PE `NIFTY18AUG26P24350` TP +14.20 points / Rs +914.57
  - 11:20 -> 12:07 PE `NIFTY18AUG26P24350` SL -0.35 points / Rs -30.56
  - 12:19 -> 12:21 CE `NIFTY18AUG26C24300` TP -0.75 points / Rs -61.49
  - 12:44 -> 13:42 CE `NIFTY18AUG26C24300` TP +16.70 points / Rs +1072.72
  - 14:12 -> 14:18 CE `NIFTY18AUG26C24350` SL -3.10 points / Rs -213.69
  - 14:55 -> 15:00 PE `NIFTY18AUG26P24400` TP +2.90 points / Rs +181.95

### `smart-fib|3m|tp0.29|sl1.25`

- 2026-08-12: 2 trades, 2 matched, net -0.40 points / Rs -48.06
  - 10:34 -> 10:48 PE `NIFTY18AUG26P24400` TP +18.60 points / Rs +1196.82
  - 12:59 -> 14:30 PE `NIFTY18AUG26P24300` SL -19.00 points / Rs -1244.88
- 2026-08-13: 2 trades, 2 matched, net +12.50 points / Rs +788.41
  - 10:49 -> 11:35 PE `NIFTY18AUG26P24400` SL -5.00 points / Rs -335.74
  - 14:27 -> 14:56 PE `NIFTY18AUG26P24500` TP +17.50 points / Rs +1124.15
- 2026-08-14: 7 trades, 7 matched, net +14.90 points / Rs +899.43
  - 10:23 -> 10:32 CE `NIFTY18AUG26C24350` SL -9.45 points / Rs -623.18
  - 10:51 -> 10:59 PE `NIFTY18AUG26P24350` TP +14.20 points / Rs +914.57
  - 11:20 -> 12:11 PE `NIFTY18AUG26P24350` SL -3.05 points / Rs -205.87
  - 12:19 -> 12:21 CE `NIFTY18AUG26C24300` TP -0.75 points / Rs -61.49
  - 12:44 -> 13:42 CE `NIFTY18AUG26C24300` TP +16.70 points / Rs +1072.72
  - 14:12 -> 14:21 CE `NIFTY18AUG26C24350` SL -5.65 points / Rs -379.27
  - 14:55 -> 15:00 PE `NIFTY18AUG26P24400` TP +2.90 points / Rs +181.95

### `smart-fib|5m|tp0.29|sl1.155`

- 2026-08-12: 2 trades, 2 matched, net +10.90 points / Rs +685.68
  - 10:34 -> 10:48 PE `NIFTY18AUG26P24400` TP +18.60 points / Rs +1196.82
  - 12:59 -> 13:19 PE `NIFTY18AUG26P24300` SL -7.70 points / Rs -511.14
- 2026-08-13: 3 trades, 3 matched, net +6.95 points / Rs +419.24
  - 10:49 -> 10:52 PE `NIFTY18AUG26P24400` SL -3.00 points / Rs -205.88
  - 11:28 -> 11:36 PE `NIFTY18AUG26P24350` SL -7.55 points / Rs -499.03
  - 14:27 -> 14:56 PE `NIFTY18AUG26P24500` TP +17.50 points / Rs +1124.15
- 2026-08-14: 7 trades, 7 matched, net +23.70 points / Rs +1470.83
  - 10:23 -> 10:31 CE `NIFTY18AUG26C24350` SL -5.90 points / Rs -392.67
  - 10:51 -> 10:59 PE `NIFTY18AUG26P24350` TP +14.20 points / Rs +914.57
  - 11:20 -> 12:07 PE `NIFTY18AUG26P24350` SL -0.35 points / Rs -30.56
  - 12:19 -> 12:21 CE `NIFTY18AUG26C24300` TP -0.75 points / Rs -61.49
  - 12:44 -> 13:42 CE `NIFTY18AUG26C24300` TP +16.70 points / Rs +1072.72
  - 14:12 -> 14:18 CE `NIFTY18AUG26C24350` SL -3.10 points / Rs -213.69
  - 14:55 -> 15:00 PE `NIFTY18AUG26P24400` TP +2.90 points / Rs +181.95

### `smart-fib|5m|tp0.29|sl1.25`

- 2026-08-12: 2 trades, 2 matched, net -0.40 points / Rs -48.06
  - 10:34 -> 10:48 PE `NIFTY18AUG26P24400` TP +18.60 points / Rs +1196.82
  - 12:59 -> 14:30 PE `NIFTY18AUG26P24300` SL -19.00 points / Rs -1244.88
- 2026-08-13: 2 trades, 2 matched, net +12.50 points / Rs +788.41
  - 10:49 -> 11:35 PE `NIFTY18AUG26P24400` SL -5.00 points / Rs -335.74
  - 14:27 -> 14:56 PE `NIFTY18AUG26P24500` TP +17.50 points / Rs +1124.15
- 2026-08-14: 7 trades, 7 matched, net +14.90 points / Rs +899.43
  - 10:23 -> 10:32 CE `NIFTY18AUG26C24350` SL -9.45 points / Rs -623.18
  - 10:51 -> 10:59 PE `NIFTY18AUG26P24350` TP +14.20 points / Rs +914.57
  - 11:20 -> 12:11 PE `NIFTY18AUG26P24350` SL -3.05 points / Rs -205.87
  - 12:19 -> 12:21 CE `NIFTY18AUG26C24300` TP -0.75 points / Rs -61.49
  - 12:44 -> 13:42 CE `NIFTY18AUG26C24300` TP +16.70 points / Rs +1072.72
  - 14:12 -> 14:21 CE `NIFTY18AUG26C24350` SL -5.65 points / Rs -379.27
  - 14:55 -> 15:00 PE `NIFTY18AUG26P24400` TP +2.90 points / Rs +181.95

### `smart-fib|combined|tp0.29|sl1.155`

- 2026-08-12: 2 trades, 2 matched, net +10.90 points / Rs +685.68
  - 10:34 -> 10:48 PE `NIFTY18AUG26P24400` TP +18.60 points / Rs +1196.82
  - 12:59 -> 13:19 PE `NIFTY18AUG26P24300` SL -7.70 points / Rs -511.14
- 2026-08-13: 3 trades, 3 matched, net +6.95 points / Rs +419.24
  - 10:49 -> 10:52 PE `NIFTY18AUG26P24400` SL -3.00 points / Rs -205.88
  - 11:28 -> 11:36 PE `NIFTY18AUG26P24350` SL -7.55 points / Rs -499.03
  - 14:27 -> 14:56 PE `NIFTY18AUG26P24500` TP +17.50 points / Rs +1124.15
- 2026-08-14: 7 trades, 7 matched, net +23.70 points / Rs +1470.83
  - 10:23 -> 10:31 CE `NIFTY18AUG26C24350` SL -5.90 points / Rs -392.67
  - 10:51 -> 10:59 PE `NIFTY18AUG26P24350` TP +14.20 points / Rs +914.57
  - 11:20 -> 12:07 PE `NIFTY18AUG26P24350` SL -0.35 points / Rs -30.56
  - 12:19 -> 12:21 CE `NIFTY18AUG26C24300` TP -0.75 points / Rs -61.49
  - 12:44 -> 13:42 CE `NIFTY18AUG26C24300` TP +16.70 points / Rs +1072.72
  - 14:12 -> 14:18 CE `NIFTY18AUG26C24350` SL -3.10 points / Rs -213.69
  - 14:55 -> 15:00 PE `NIFTY18AUG26P24400` TP +2.90 points / Rs +181.95

### `smart-fib|combined|tp0.29|sl1.25`

- 2026-08-12: 2 trades, 2 matched, net -0.40 points / Rs -48.06
  - 10:34 -> 10:48 PE `NIFTY18AUG26P24400` TP +18.60 points / Rs +1196.82
  - 12:59 -> 14:30 PE `NIFTY18AUG26P24300` SL -19.00 points / Rs -1244.88
- 2026-08-13: 2 trades, 2 matched, net +12.50 points / Rs +788.41
  - 10:49 -> 11:35 PE `NIFTY18AUG26P24400` SL -5.00 points / Rs -335.74
  - 14:27 -> 14:56 PE `NIFTY18AUG26P24500` TP +17.50 points / Rs +1124.15
- 2026-08-14: 7 trades, 7 matched, net +14.90 points / Rs +899.43
  - 10:23 -> 10:32 CE `NIFTY18AUG26C24350` SL -9.45 points / Rs -623.18
  - 10:51 -> 10:59 PE `NIFTY18AUG26P24350` TP +14.20 points / Rs +914.57
  - 11:20 -> 12:11 PE `NIFTY18AUG26P24350` SL -3.05 points / Rs -205.87
  - 12:19 -> 12:21 CE `NIFTY18AUG26C24300` TP -0.75 points / Rs -61.49
  - 12:44 -> 13:42 CE `NIFTY18AUG26C24300` TP +16.70 points / Rs +1072.72
  - 14:12 -> 14:21 CE `NIFTY18AUG26C24350` SL -5.65 points / Rs -379.27
  - 14:55 -> 15:00 PE `NIFTY18AUG26P24400` TP +2.90 points / Rs +181.95
