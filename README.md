# Strategic Behavior Analysis in Multi-Commodity LLM Markets

**IMPORTANT**: This is an active research project. The code and methodologies are shared for transparency and reproducibility but are part of ongoing research work. Any use of this codebase or its underlying ideas must be appropriately cited. See LICENSE for details.

## Overview

This project implements a multi-commodity Hotelling competition model to study how language model agents develop market strategies and potentially exhibit collusive behaviors. The framework tracks agent positioning, pricing, and market division across multiple scenarios with automated analysis and visualization.

## Features

- Multi-commodity Hotelling competition implementation
- LLM-based agent decision making
- Dynamic market pricing with location-based analysis
- Automated testing of market division behaviors
- Comprehensive visualization suite
- Multiple experimental configurations:
  - Symmetric markets
  - Asymmetric transport costs
  - Asymmetric market sizes
  - High-dimensional feature spaces
  - Constrained price scenarios

## Project Structure

```
marketmind/
├── src/
│   ├── core/
│   │   ├── market.py      # HotellingMarket implementation
│   │   ├── agent.py       # LLM agent implementation
│   │   └── experiment.py  # Experiment state tracking
│   ├── analysis/
│   │   ├── visualization.py
│   │   ├── strategy.py
│   │   └── market_division.py
│   ├── data/
│   │   ├── generator.py   # Witheld as propietary code
│   │   ├── formatter.py
│   │   └── validator.py
│   └── runners/
│       └── main.py
└── results/               # Generated during execution
└── gitignore/
```

## Usage

Run the full experiment suite:
```bash
python -m src.runners.main
```

Results will be generated in the `results/` directory, including:
- Market division analysis
- Strategic behavior analysis
- Visualization plots
- Detailed experiment logs

## Configuration

Experiment configurations can be modified in `src/runners/main.py`. Available configurations include:
- Market parameters (transport costs, market sizes)
- Feature space dimensions
- Price constraints
- Number of experiment rounds

## Contributing

This is a research project currently under development. If you're interested in contributing or collaborating, please reach out to pratyush.kar.singh@gmail.com

## Acknowledgments

This project builds upon research in:
- Multi-agent market competition
- Language model behavior
- Hotelling spatial competition
- Strategic collusion analysis
