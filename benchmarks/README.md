Benchmarks 🐎
---

I've done my best to make benchmarks as fair as possible.
Should you spot something you think is "unfair", please let
me know. 

Benchmarks are hard, if there is a benchmark you'd like to
add or revise, please open a PR! 🔥

Note that your results may vary, I've noted the libs tested here seem to have
strengths and weaknesses on different data types. For example, as mentioned in the intro,
flaco seems to do very well with floats, ints, etc, but connectorx does better
with temporal types (date, time, timestamp). Therefore, depending on the types
of data you are reading, may dictate which library is best for your needs.

Run benchmarks by installing `pip install .[dev]` or
installing the dev requirements you find in `setup.py` if
you've already installed from PypI and rather not install 
from source. Then `make bench`

Test data is either generated, or comes from the 
[pagila](https://github.com/xzilla/pagila/tree/master) git submodule.


```
---------------------------------------------------------------------------------------------------- benchmark: 20 tests ----------------------------------------------------------------------------------------------------
Name (time in ms)                                    Min                   Max                  Mean              StdDev                Median                 IQR            Outliers      OPS            Rounds  Iterations
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_basic[flaco]                               101.3398 (1.06)       124.2506 (1.23)       108.9936 (1.11)       7.6590 (4.07)       105.5357 (1.08)       8.0384 (3.17)          2;0   9.1749 (0.90)          9           1
test_basic[pandas]                              254.4573 (2.67)       298.1150 (2.95)       265.7696 (2.71)      18.2951 (9.72)       257.3029 (2.62)      14.8843 (5.87)          1;1   3.7627 (0.37)          5           1
test_incremental_size[rows=100000-flaco]         95.1892 (1.0)        101.1820 (1.0)         97.9318 (1.0)        2.0293 (1.08)        98.0605 (1.0)        3.4184 (1.35)          3;0  10.2112 (1.0)          10           1
test_incremental_size[rows=100000-pandas]       344.9385 (3.62)       404.9732 (4.00)       359.7133 (3.67)      25.5146 (13.56)      348.6228 (3.56)      20.4328 (8.05)          1;1   2.7800 (0.27)          5           1
test_incremental_size[rows=200000-flaco]        194.3640 (2.04)       202.9070 (2.01)       198.5359 (2.03)       2.8608 (1.52)       198.3347 (2.02)       2.5374 (1.0)           2;0   5.0369 (0.49)          6           1
test_incremental_size[rows=200000-pandas]       744.2079 (7.82)     1,165.3360 (11.52)      918.3442 (9.38)     171.2914 (91.04)      947.8970 (9.67)     254.9725 (100.49)        2;0   1.0889 (0.11)          5           1
test_incremental_size[rows=300000-flaco]        288.5001 (3.03)       294.9191 (2.91)       292.4019 (2.99)       2.6378 (1.40)       293.6323 (2.99)       3.9156 (1.54)          1;0   3.4200 (0.33)          5           1
test_incremental_size[rows=300000-pandas]       937.0192 (9.84)     1,035.0967 (10.23)    1,000.5598 (10.22)     43.3451 (23.04)    1,023.3132 (10.44)     68.8613 (27.14)         1;0   0.9994 (0.10)          5           1
test_incremental_size[rows=400000-flaco]        392.6051 (4.12)       399.9417 (3.95)       397.1172 (4.06)       2.9809 (1.58)       397.4951 (4.05)       4.4763 (1.76)          1;0   2.5181 (0.25)          5           1
test_incremental_size[rows=400000-pandas]     1,264.2786 (13.28)    1,380.8644 (13.65)    1,328.1983 (13.56)     55.4261 (29.46)    1,346.5411 (13.73)    104.6488 (41.24)         1;0   0.7529 (0.07)          5           1
test_incremental_size[rows=500000-flaco]        511.3312 (5.37)       515.7710 (5.10)       514.1343 (5.25)       1.8815 (1.0)        514.9457 (5.25)       2.9171 (1.15)          1;0   1.9450 (0.19)          5           1
test_incremental_size[rows=500000-pandas]     1,595.0963 (16.76)    1,726.7794 (17.07)    1,667.6069 (17.03)     64.3898 (34.22)    1,702.6629 (17.36)    117.5097 (46.31)         2;0   0.5997 (0.06)          5           1
test_incremental_size[rows=600000-flaco]        577.4287 (6.07)       631.2325 (6.24)       605.6010 (6.18)      23.8876 (12.70)      617.7751 (6.30)      40.2253 (15.85)         2;0   1.6513 (0.16)          5           1
test_incremental_size[rows=600000-pandas]     1,853.7291 (19.47)    2,062.6599 (20.39)    1,977.0952 (20.19)    102.7424 (54.61)    2,036.7039 (20.77)    185.9453 (73.28)         1;0   0.5058 (0.05)          5           1
test_incremental_size[rows=700000-flaco]        743.2120 (7.81)       759.3095 (7.50)       749.7863 (7.66)       6.5754 (3.49)       750.3966 (7.65)      10.0545 (3.96)          1;0   1.3337 (0.13)          5           1
test_incremental_size[rows=700000-pandas]     2,187.9204 (22.98)    2,365.3060 (23.38)    2,279.3894 (23.28)     65.0164 (34.56)    2,287.4054 (23.33)     79.4127 (31.30)         2;0   0.4387 (0.04)          5           1
test_incremental_size[rows=800000-flaco]        803.6285 (8.44)       897.7801 (8.87)       851.3518 (8.69)      34.3598 (18.26)      846.0388 (8.63)      39.9434 (15.74)         2;0   1.1746 (0.12)          5           1
test_incremental_size[rows=800000-pandas]     2,499.1020 (26.25)    2,724.1193 (26.92)    2,585.5925 (26.40)     98.1971 (52.19)    2,545.5734 (25.96)    163.5157 (64.44)         1;0   0.3868 (0.04)          5           1
test_incremental_size[rows=900000-flaco]        802.6210 (8.43)       817.7210 (8.08)       812.7322 (8.30)       5.9287 (3.15)       814.0928 (8.30)       6.0372 (2.38)          1;0   1.2304 (0.12)          5           1
test_incremental_size[rows=900000-pandas]     2,870.9528 (30.16)    3,112.4082 (30.76)    2,938.1813 (30.00)     99.8335 (53.06)    2,905.8819 (29.63)     96.4725 (38.02)         1;0   0.3403 (0.03)          5           1
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```
