# Power Market Anomaly Detection

A PyTorch-based deep learning system for detecting anomalies in power market data, designed for integration into power market modeling workflows. The system employs a novel TiFe (Time-Feature) attention mechanism combined with encoder-decoder architecture to identify unusual patterns in time series power market data.

## Overview

This project implements an attention-based anomaly detection model specifically tailored for power market applications. The architecture leverages parallel training capabilities and provides interpretable attention maps that enable domain experts to understand the source and nature of detected anomalies.

## Motivation

The approach is inspired by research demonstrating the effectiveness of attention mechanisms in time series anomaly detection. Key advantages include:

- **Parallel Training**: Attention mechanisms enable efficient parallel computation during training
- **Interpretability**: Attention maps provide transparent insights into model decision-making
- **Domain Applicability**: Designed specifically for the unique characteristics of power market data

Reference: [Attention-based anomaly detection in time series](https://www.sciencedirect.com/science/article/pii/S2666546823000642)

## Architecture

The model implements a three-stage architecture:

### 1. TiFe Attention Mechanism

The core innovation is a custom attention layer that operates simultaneously across multiple dimensions:

- **Time Attention**: Attends across temporal sequences to capture time-dependent patterns
- **Feature Attention**: Attends across feature dimensions to identify cross-variable relationships  
- **Window Attention**: Processes relationships between different time windows in batches
- **ANN Extraction**: Neural network layer that transforms concatenated representations back to original dimensions

**Input**: `(batch_size, window_size, num_features)`  
**Processing**: Separate attention computations across time and feature dimensions  
**Output**: `(batch_size, window_size, num_features)` with attention maps for interpretability

### 2. Encoder-Decoder Architecture

A bottleneck architecture for anomaly detection through reconstruction:

- **Encoder**: 3-layer compression network reducing features to bottleneck representation
- **Bottleneck**: Compressed hidden representation with reduced dimensionality
- **Decoder**: 3-layer reconstruction network restoring original feature dimensions
- **Regularization**: Dropout and ReLU activations throughout the network

### 3. Final TiFe Attention Layer

Second attention processing stage that refines reconstructed representations and provides additional interpretability through attention map analysis.

## Data Sources

### ERCOT (Electric Reliability Council of Texas)

Primary data source selected for transparency and data availability from [ERCOT Data Portal](https://data.ercot.com/).

**Data Types**:
- **Zonal LMP**: 5-minute resolution by load zone, aggregated to hourly
- **Actual Demand**: Hourly resolution by forecast zone  
- **Forecasted Demand**: Hourly resolution by forecast zone
- **Wind Production**: Hourly resolution by region
- **Solar Production**: Hourly resolution by region

Future versions may incorporate additional ISO/RTO data sources as they become available.

## Installation

The project uses `uv` for dependency management:

```bash
# Install dependencies
uv sync

# Run the main application  
python main.py

# Alternative execution with uv
uv run python main.py
```

## Requirements

- Python >= 3.11
- PyTorch >= 2.7.1

## Project Structure

```
src/models/
├── tife_attention.py                 # TiFe attention layer implementation
└── power_market_anomaly_detector.py  # Main detector architecture

data/ercot/
├── ercot_data.py                     # ERCOT data utilities and processing

tests/                                # Test suite (in development)
```

## Usage

### Model Initialization

```python
from src.models.power_market_anomaly_detector import AnomalyDetector

model = AnomalyDetector(
    window_size=24,      # Hours in sequence
    num_features=5,      # Number of input features  
    hidden_dim=64,       # Hidden layer dimensionality
    batch_size=32,       # Batch size
    dropout=0.1          # Dropout rate
)
```

### Anomaly Detection

```python
# Forward pass returns reconstructed output, anomaly scores, reconstruction error, and attention maps
reconstructed, anomaly_scores, reconstruction_error, attention_maps = model(input_data)

# Binary anomaly detection with threshold
anomalies = model.detect_anomalies(input_data, threshold=0.5)

# Training loss computation
loss = model.get_reconstruction_loss(input_data)
```

### Attention Analysis

The model provides interpretable attention maps for analysis:

```python
# Access attention maps from both attention layers
initial_attention = attention_maps['initial_attention']
final_attention = attention_maps['final_attention']

# Individual attention components
time_attention = initial_attention['time_attention']
feature_attention = initial_attention['feature_attention'] 
window_attention = initial_attention['window_attention']
```

## Evaluation and Results

**Note**: This project is currently under development. Full evaluation results will be provided upon completion.

### Planned Evaluation Metrics

- **Reconstruction Accuracy**: Model's ability to reconstruct normal power market patterns
- **Anomaly Detection Performance**: Precision, recall, and F1-score on labeled anomalous events
- **Attention Map Analysis**: Visualization of attention patterns during normal vs. anomalous periods
- **Temporal Analysis**: Model performance across different time scales and market conditions

### Expected Outputs

- Quantitative reconstruction performance compared to actual market data
- Attention map visualizations highlighting model focus during anomaly detection
- Comparative analysis of attention patterns between normal and anomalous market conditions

## Development Status

This project is in active development. Key components implemented:

- [x] TiFe attention mechanism
- [x] Encoder-decoder architecture  
- [x] Anomaly detection framework
- [ ] ERCOT data integration utilities
- [ ] Training pipeline
- [ ] Evaluation framework
- [ ] Comprehensive testing suite
- [ ] Results analysis and visualization

## License

[License information to be added]

## Contributing

[Contributing guidelines to be added]
