# Prague Air Quality Hub

> Final project for *Data Processing in Python* — Institute of Economic Studies, Charles University

This project examines air quality patterns in Prague and links them to environmental and meteorological conditions. Data were retrieved from public APIs provided by the Czech Hydrometeorological Institute (CHMI), covering air quality and weather stations across the city.

The analysis is delivered in two components: an interactive Streamlit web application and a supplementary Jupyter notebook.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Academic Transparency & AI Disclosure](#academic-transparency--ai-disclosure)

---

## Overview

The project investigates spatio-temporal patterns of urban air pollution in Prague, combining historical exploratory analysis with a machine-learning simulation layer. A Random Forest Regressor model is trained on meteorological and pollution data to allow forecasting of pollutant concentrations under custom atmospheric conditions.

---

## Features

### 1. Real-Time Scenario Simulator

- **Interactive sliders** for temperature, wind speed, wind direction, humidity, pressure, and precipitation — forecasts pollutant values dynamically using the trained model.
- **Geospatial sensor network map** built on Pydeck: displays all active Prague monitoring stations, with the selected station highlighted as a red beacon.
- **EU limit diagnostics**: simulated pollutant values are compared against official European Union environmental thresholds, with warnings triggered when limits are approached or exceeded.

### 2. Historical Visualisations Workspace

- **Weather & pollution co-movements**: time-series overlays of meteorological variables and pollutant concentrations over customisable rolling calendar windows.
- **Cross-district comparison**: simultaneous visualisation of up to 16 monitoring stations to identify structural pollution differences across Prague's urban sectors.
- **Wind rose charts**: directional analysis of pollutant dispersion illustrating how wind vector distributions physically displace emissions across the city.

### 3. Jupyter Notebook — Exploratory Analysis

A self-contained notebook delivering additional graphical analysis of temporal and spatial patterns in air quality and weather, including station-level summaries and pollutant co-movement plots.

---

## Project Structure

```
├── data/
│   ├── raw/  
│   └── processed/
│       └── processed_data.csv       # Cleaned, aggregated, and synchronised dataset
├── docs                             # Miscellaneous supplementary materials
├── notebooks                        # Notebooks that were used during drafting the code
│   ├── 01_Final_Analysis.ipynb      # Exploratory data analysis notebook          
│   └── other
├── results/
│   └── models/                      # Saved Random Forest model bundle
├── src/
│   ├── data_processing.py
│   ├── download_data.py
│   ├── model_wrapper.py             
│   └── train_pipeline.py
├── app.py                           # Streamlit web application
├── run_pipeline.py                  # Data download, processing and model training pipeline
├── README.md
└── requirements.txt
```

---

## Installation

**Requirements:** Python 3.10+

```bash
# 1. Clone the repository
git clone https://github.com/jayjay7788/Data-Processing-in-Python---Project.git
cd Data-Processing-in-Python---Project

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Run the data pipeline (data processing + model training)

```bash
python run_pipeline.py
```

This script downloads and processes the raw data and trains the Random Forest model. Outputs are saved to `data/processed/` and `results/models/`. Preprocessed data and pretrained model results are provided for easier usage. Beware that the data download might take up to 1 hour due to slow download of air quality data (the server is sometimes unstable and the data are not provided in ideal format). You can select to skip all the steps in the pipeline and reuse prepared data and model results. 

### Launch the Streamlit application

```bash
streamlit run app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`) in your browser.

### Open the exploratory notebook

```bash
jupyter notebook notebooks/01_Final_Analysis.ipynb
```

---

## Data

Data were sourced from the **Czech Hydrometeorological Institute (CHMI)** via their public API. The dataset covers air quality measurements (PM2.5, PM10, NO₂, O₃, and others) and meteorological readings (temperature, humidity, wind speed/direction, pressure, precipitation) from stations located across Prague.

The processed dataset (`data/processed/processed_data.csv`) is the cleaned and aggregated version used for all analysis and model training.

---

## Academic Transparency & AI Disclosure

This project was submitted as a final assignment for the *Data Processing in Python* course at the Institute of Economic Studies, Charles University.

**Generative AI Use Statement:**
Generative AI was used in this project for troubleshooting, idea refinement, and grammar checks in accordance with course guidelines. The project topic, analytical approach, data pipeline design, and application structure were conceived and planned entirely by the authors. AI tools were also consulted during implementation to help translate our own designs into working code in cases where we understood the intended logic but needed assistance with syntax or library-specific implementation details.
