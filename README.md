# Ferguson Wildfire Analytics Pipeline

A modular, production-grade Python pipeline for large-scale wildfire data
compression, fusion and assimilation built from real simulation and satellite
data of the Ferguson wildfire (California, 2018).


---

## What This Project Does

This pipeline processes large wildfire datasets through five analytical stages:

```
Raw Data (4GB+)
      ↓
Linear Compression      →  TruncatedSVD, 114 modes, 95% variance retained
      ↓
Nonlinear Compression   →  UNet Autoencoder with skip connections
      ↓
Data Fusion             →  Latent space blending of two observations
      ↓
Assimilation (Linear)   →  Kalman filter in TSVD latent space
      ↓
Assimilation (NL)       →  Kalman filter in AE latent space
      ↓
Results                 →  outputs/ (local) or Cosmos DB (Azure)
```

The pipeline is **environment-agnostic** - it runs locally out of the box and
switches to Azure Blob Storage + Cosmos DB by changing one environment variable.

---

## Why This Architecture

The core design decision was to separate the analytics pipeline from the storage
layer entirely. The pipeline never knows or cares whether data lives on local
disk or in Azure Blob Storage - it just calls `storage.load_array()` and
`storage.save_results()`.

This means moving to Azure requires no changes to any analytical code:

```bash
export ENVIRONMENT=azure
export AZURE_STORAGE_CONNECTION_STRING=...
export COSMOS_CONNECTION_STRING=...
python pipeline.py
```

---

## Key Technical Decisions

### Linear Compression
- Used **TruncatedSVD** over IncrementalPCA - 28s fit vs 135s on same data
- Memory-mapped training data - never loads the full array into RAM
- 114 components captures 95% of variance on sparse binary fire masks

### Nonlinear Compression
- **UNet autoencoder** with skip connections - preserves sharp fire boundaries
  that a plain deep AE blurs out
- Custom **Sobel edge loss** on top of MSE - explicitly penalises blurry edges
- Early stopping with checkpointing - best weights reloaded after training

### Data Assimilation
- Kalman filter runs in **latent space**, not pixel space
- 65,536-dimensional problem reduced to 114 dimensions before update
- Full empirical covariance matrix **B** - correlated modes share information
- **β** tuned by validation sweep - 0.68 gave optimal MSE
- Nonlinear AE assimilation achieved **15× MSE improvement** over background

---

## Results

| Stage | MSE | Time |
|---|---|---|
| Linear compression | 5.74e-03 | 28s fit / 9s reconstruct |
| Nonlinear compression | ~1e-05 | ~15s reconstruct |
| Assimilation (linear) | ~44% improvement over background | <1s update |
| Assimilation (nonlinear) | ~15× improvement over background | <1s update |

---

## Project Structure

```
ferguson-wildfire-analytics/
│
├── pipeline.py                  # master orchestrator - runs all stages
├── config.py                    # all settings + storage factory
├── requirements.txt
├── requirements-azure.txt
│
├── src/
│   ├── storage/
│   │   ├── base.py              # abstract storage interface
│   │   ├── local.py             # local filesystem implementation
│   │   └── azure_blob.py        # Azure Blob + Cosmos DB 
│   ├── compression/
│   │   ├── tsvd.py              # linear compression
│   │   └── autoencoder.py       # nonlinear compression
│   ├── fusion/
│   │   └── latent_fusion.py     # latent space data fusion
│   ├── assimilation/
│   │   ├── kalman_linear.py     # Kalman filter with TSVD
│   │   └── kalman_nonlinear.py  # Kalman filter with AE
│   └── utils/
│       ├── logging_config.py
│       ├── metrics.py
│       └── visualisation.py
│
├── scripts/
│   ├── run_compression.py
│   ├── run_encoding.py
│   ├── run_fusion.py
│   ├── run_assimilation_linear.py
│   └── run_assimilation_nonlinear.py
│
├── tests/
│   ├── test_compression.py
│   └── test_assimilation.py
│
└──.github/
    └── workflows/
        └── run_tests.yml
```

---

## Installation

```bash
git clone https://github.com/ansinha27/ferguson-wildfire-analytics.git
cd ferguson-wildfire-analytics

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

## Usage

### Run the full pipeline
```bash
python pipeline.py
```

### Run a single stage
```bash
python pipeline.py --stage compression
python pipeline.py --stage assimilation_linear
```

### Run individual scripts
```bash
python scripts/run_compression.py --n_components 114
python scripts/run_assimilation_linear.py --beta 0.68
```

### Skip a stage
```bash
python pipeline.py --skip encoding
```

### Switch to Azure
```bash
export ENVIRONMENT=azure
export AZURE_STORAGE_CONNECTION_STRING=your_string
export COSMOS_CONNECTION_STRING=your_string
pip install -r requirements-azure.txt
python pipeline.py
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Skills Demonstrated

| Area | Detail |
|---|---|
| **Large data handling** | Memory-mapped arrays, batched processing, never loading full dataset into RAM |
| **Cloud-agnostic design** | Storage abstraction layer - swap local for Azure with one config change |
| **Deep learning** | UNet architecture, custom Sobel edge loss, early stopping, GPU/CPU portable |
| **Data assimilation** | Kalman filter, empirical covariance estimation, latent space methods |
| **Software engineering** | SOLID principles, type hints, structured logging, CI/CD, unit testing |
