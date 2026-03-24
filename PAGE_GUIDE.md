# GridIntel Application: Workflow Ladder Guide

This document outlines the structured analytical ladder within the GridIntel platform. Each stage builds progressively, moving from foundational diagnostics to actionable forecasting decisions.

---

## 1. Load Analysis (`load-analysis`)

The **Load Analysis** layer is the foundational diagnostic stage. It focuses on understanding demand behavior before introducing external drivers.

### Core Capabilities

**96-Block Load Visualization**

* Historical vs recent load curves
* Peak and trough identification
* Plateau and ramp detection

**Pattern Decomposition**

* Morning ramp intensity
* Midday plateau behavior
* Evening peak sharpness
* Night tail decay

**Anomaly Detection**

* Sudden dips / spikes
* Structural breaks
* Outage-linked deviations

**Trend Signals**

* Day-over-day momentum
* Weekly seasonality
* Growth normalization

This stage answers:

> “What is the grid doing on its own?”

---

## 2. Weather Analysis (`weather-analysis`)

Once load structure is understood, the next layer attributes atmospheric influence.

### Weather Driver Mapping

**State-Level Weather Synthesis**

* Temperature aggregation
* Humidity stress mapping
* Rainfall intensity overlays
* Cloud and solar suppression signals

**Impact Attribution**

* Cooling degree load impact
* Humid heat amplification
* Rain suppression dips
* Wind cooling elasticity

**Driver Interaction Views**

* Temp × Humidity stress index
* Rain × Evening peak suppression
* Cloud × Midday solar dips

**Regime Classification**

* Heatwave
* Humid heat
* Rainy suppression
* Transition weather

This stage answers:

> “How is the atmosphere bending the load curve?”

---

## 3. Simulator — Corrections Workspace (`dayahead`)

The **Simulator** is the intervention layer where analysts actively refine the forecast.

### Forecast Editing Environment

**96-Block Forecast Simulator**

* Baseline vs adjusted forecast
* Overlay with actuals (if available)

**Driver Corrections**

Weather Overrides:

* Temperature adjustments
* Humidity shifts
* Rainfall injection/removal

Bias Correction:

* Automatic bias detection
* Manual MW correction

Calendar Configuration:

* Weekday / weekend switching
* Holiday profile injection

**Granular Editing**

Block Grid Editor:

* MW edits per 15-min block
* Direct curve sculpting

Chart Interaction:

* Shift + drag block selection
* Range-level adjustments

**Scenario Management**

* Save forecast versions
* Scenario comparison modal

This stage answers:

> “What corrections are needed before dispatch?”

---

## 4. Integrated Analysis (`analysis`)

After simulation, the **Analysis** layer evaluates forecast intelligence and model behavior.

### KPI Intelligence Library

Forecast Accuracy:

* MAPE
* Peak accuracy
* Block RMSE

Attribution Diagnostics:

* Weather impact share
* Calendar effect magnitude
* Bias persistence

Load Shape Metrics:

* Ramp accuracy
* Plateau deviation
* Tail decay error

Operational Decision Quality:

* Dispatch alignment
* Risk classification

---

### Impact Matrices

**Driver Attribution Matrix**

* MW contribution by driver:

  * Temperature
  * Humidity
  * Rain
  * Cloud
* Blockwise decomposition

**Causal Root Cause Engine**
Errors classified into:

* Weather-driven
* System-driven
* Model-driven

---

### Variance Visualization

* Heatmaps of forecast error
* Blockwise deviation bars
* Peak risk zones

This stage answers:

> “Why did the forecast behave this way?”

---

## 5. Forecast & Decision Layer (`forecast`)

The final stage converts corrected and validated intelligence into operational forecasts.

### Forecast Outputs

**Final Day-Ahead Forecast**

* Corrected 96-block curve
* Bias-adjusted projection
* Weather-aligned load

**Confidence Scoring**

* Reliability index
* Regime risk factor
* Scenario spread width

**Probabilistic Bands**

* P10 (cool scenario)
* P50 (base case)
* P90 (heatwave risk)

**Reserve Signals**

* Peak buffer requirement
* Dip risk alert
* Ramp adequacy

---

### Dispatch Intelligence

* Decision signals
* Corrective dispatch triggers
* Load risk ranking

This stage answers:

> “What should the grid prepare for?”

---

# Ladder Summary

| Stage            | Focus               | Key Question        |
| ---------------- | ------------------- | ------------------- |
| Load Analysis    | Demand structure    | What is load doing? |
| Weather Analysis | Atmospheric drivers | Why is it doing it? |
| Simulator        | Corrections         | What should change? |
| Analysis         | Diagnostics         | How did we perform? |
| Forecast         | Decisions           | What happens next?  |
