# Inference Speed Benchmark Results

**CPU Model:** Intel(R) Pentium(R) Silver N5030 CPU @ 1.10GHz | **RAM:** 8.00 GB

### Baseline Inference Throughput (`results/speed_results.csv`)

| Batch Size | Mean (events/sec) | Min (events/sec) | Max (events/sec) |
|:---:|:---:|:---:|:---:|
| 100 | 1,744.30 | 1,672.38 | 1,795.11 |
| 1,000 | 15,178.93 | 14,953.58 | 15,375.70 |
| 10,000 | 93,988.85 | 85,862.85 | 105,805.10 |

### Feature Ablation Inference Throughput (`results/ablation_speed_results.csv`)

| Batch Size | Mean (events/sec) | Min (events/sec) | Max (events/sec) |
|:---:|:---:|:---:|:---:|
| 100 | 1,855.60 | 1,771.37 | 2,137.61 |
| 1,000 | 15,961.05 | 14,590.00 | 17,721.10 |
| 10,000 | 101,380.96 | 94,343.54 | 106,800.87 |

> **Note:** This benchmark measures model scoring throughput only (in-memory `predict_proba` inference). It does not measure end-to-end network packet capture, stream ingestion, feature computation, or disk persistence overhead.
