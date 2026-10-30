# biketrips

Predicting shared-bike demand from urban features in Shenzhen, with honest evaluation. The original tutorial trained on 1% of trips, dropped every cell with zero demand, and scored models with a random split that leaks spatial information. This repo fixes all three and asks which model actually predicts demand in neighborhoods it has never seen.

## Credit

**Based on:** *Predicting Mobility Demand from Urban Features*, Climate Change AI Summer School 2023, by Konstantin Klemmer, Shafat Rahman, Felix Wagner, Florian Nachtigall, and Gabriela Yaulli Herrera.

- Original repo: https://github.com/climatechange-ai-tutorials/mobility-demand
- DOI: https://doi.org/10.5281/zenodo.11619223
- Data: Shenzhen bike-share trip data and AMap POI data, as distributed with the tutorial.
- The original notebook is kept unchanged at `notebooks/00_original_tutorial.ipynb` (MIT license, see `LICENSE`).

## Status

Work in progress.

## Setup

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
python scripts/download_data.py
```

## Citation

```bibtex
@misc{klemmer2026predicting,
  title={Predicting Mobility Demand from Urban Features},
  author={Klemmer, Konstantin and Rahman, Shafat and Wagner, Felix and Nachtigall, Florian and Herrera, Gabriela Yaulli},
  year={2026},
  organization={Climate Change AI},
  type={Tutorial},
  doi={https://doi.org/10.5281/zenodo.11619223},
  booktitle={Climate Change AI Summer School},
  howpublished={\url{https://github.com/climatechange-ai-tutorials/mobility-demand}}
}
```
