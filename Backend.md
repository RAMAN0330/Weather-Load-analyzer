# Load Simulation Application - Grid Operations KPIs
## Day-Ahead Forecast (96 Time Blocks)

---

## 1. FORECAST ACCURACY KPIs

### KPI 1: Mean Absolute Percentage Error (MAPE)
- **Formula:** `(1/96) × Σ|(Actual - Forecast) / Actual| × 100`
- **Target:** < 3% for day-ahead forecast
- **Industry Benchmark:** 2-5% for mature systems
- **Measurement:** Overall 96-block average
- **Reporting:** Daily, with 7-day and 30-day rolling averages

### KPI 2: Peak Load Forecast Accuracy
- **Formula:** `|Actual Peak - Forecasted Peak| / Actual Peak × 100`
- **Target:** < 2%
- **Critical Threshold:** > 3% triggers investigation
- **Absolute Error Target:** < 1,000 kW
- **Business Impact:** Determines capacity commitment costs

### KPI 3: Daily Energy Forecast Error
- **Formula:** `|Actual Daily kWh - Forecasted Daily kWh| / Actual Daily kWh × 100`
- **Target:** < 2.5%
- **Business Impact:** Energy procurement cost variance
- **Reporting:** Daily with monthly cost impact summary

### KPI 4: Root Mean Square Error (RMSE)
- **Formula:** `√[(1/96) × Σ(Actual - Forecast)²]`
- **Target:** < 500 kW per time block
- **Advantage:** Penalizes large errors more heavily
- **Use Case:** Identifies systematic forecast problems

### KPI 5: Forecast Bias (Systematic Error)
- **Formula:** `(1/96) × Σ(Forecast - Actual)`
- **Target:** ±100 kW (close to zero)
- **Positive Value:** Systematic over-forecasting
- **Negative Value:** Systematic under-forecasting
- **Action:** Triggers model calibration

### KPI 6: Time Block Accuracy Distribution
- **Metric:** Percentage of 96 blocks within error thresholds
- **Targets:**
  - Within ±2%: > 70% of blocks
  - Within ±5%: > 90% of blocks
  - > ±10%: < 2% of blocks
- **Use:** Identifies problematic time periods

---

## 2. LOAD ATTRIBUTION & ROOT CAUSE KPIs

### KPI 7: Weather Impact Attribution - Temperature
- **Metric:** Load change attributed to temperature deviation from comfort zone
- **Formula:** `Weather Load Impact = (Simulated with Actual Weather) - (Baseline Normal Weather)`
- **Breakdown:**
  - Cooling Load (Temperature > 22°C): kW and % of total variance
  - Heating Load (Temperature < 15°C): kW and % of total variance
  - Neutral Zone (15-22°C): Minimal impact
- **Sensitivity Coefficient:** kW per °C deviation
- **Display:** "Temperature added +2,450 kW (65%) to load variance"

### KPI 8: Weather Impact Attribution - Humidity
- **Metric:** Load change attributed to humidity variation
- **Formula:** `Humidity Impact = Load Delta × Humidity Modifier`
- **Typical Impact:** 10-20% of temperature impact
- **Combined Effect:** High temp + high humidity = amplified cooling load
- **Display:** "Humidity added +380 kW (10%) to load variance"

### KPI 9: Weather Impact Attribution - Precipitation
- **Metric:** Load reduction during precipitation events
- **Formula:** `Precip Impact = Baseline Load × Precipitation Factor × Duration`
- **Typical Range:** 2-5% reduction during daytime, minimal at night
- **Display:** "Rainfall reduced load by -620 kW (15%) during peak hours"

### KPI 10: Calendar Effect Attribution
- **Metric:** Load change attributed to day type
- **Breakdown:**
  - Weekday vs Weekend Impact: % difference
  - Holiday Impact: % reduction from typical weekday
  - Day-After-Holiday Effect: % change
- **Display:** "Weekend pattern reduced load by 8,500 kWh (18%) vs weekday baseline"

### KPI 11: Combined Attribution Analysis
- **Metric:** Decompose total load variance into contributing factors
- **Components:**
  - Temperature Impact: X kW (Y%)
  - Humidity Impact: X kW (Y%)
  - Precipitation Impact: X kW (Y%)
  - Calendar Effect: X kW (Y%)
  - Unexplained Variance: X kW (Y%)
- **Target:** Explained variance > 85%
- **Display:** Waterfall chart or stacked bar showing each contribution

---

## 3. LOAD PATTERN ANALYSIS KPIs

### KPI 12: Load Dip Analysis
- **Metric:** Identification and attribution of load decreases
- **Detection:** Load drop > 5% vs baseline for same time block
- **Analysis:**
  - **Time of Dip:** Block number and hour
  - **Magnitude:** kW and % decrease
  - **Duration:** Number of consecutive blocks
  - **Primary Cause:**
    - Weather: Precipitation, temperature drop, cloud cover
    - Calendar: Weekend, holiday, post-holiday
    - Operational: Planned outage, demand response
    - Unexplained: Requires investigation
- **Example Output:**
  ```
  Dip Detected: Block 48-52 (12:00-13:00)
  Magnitude: -3,200 kW (-12.8%)
  Duration: 5 blocks (75 minutes)
  Attribution:
    - Heavy rainfall: -2,100 kW (65.6%)
    - Temperature drop: -800 kW (25.0%)
    - Unexplained: -300 kW (9.4%)
  ```

### KPI 13: Load Rise Analysis
- **Metric:** Identification and attribution of load increases
- **Detection:** Load increase > 5% vs baseline for same time block
- **Analysis:**
  - **Time of Rise:** Block number and hour
  - **Magnitude:** kW and % increase
  - **Duration:** Number of consecutive blocks
  - **Primary Cause:**
    - Weather: High temperature, high humidity
    - Calendar: Return from holiday/weekend
    - Event: Special event, sports game
    - Unexplained: Requires investigation
- **Example Output:**
  ```
  Rise Detected: Block 68-76 (17:00-19:00)
  Magnitude: +4,800 kW (+18.2%)
  Duration: 9 blocks (135 minutes)
  Attribution:
    - High temperature (32°C): +3,400 kW (70.8%)
    - High humidity (85%): +900 kW (18.8%)
    - Evening peak pattern: +300 kW (6.3%)
    - Unexplained: +200 kW (4.2%)
  ```

### KPI 14: Ramp Rate (Load Change Velocity)
- **Metric:** Speed of load changes between consecutive blocks
- **Formula:** `Ramp Rate = |Load[block] - Load[block-1]| / 15 minutes`
- **Critical Threshold:** > 1,000 kW/15min
- **Use Case:** Generator ramping capability planning
- **Attribution Analysis:**
  - Weather-driven ramps (temperature change, rain start/stop)
  - Behavioral ramps (work start/end, appliance cycling)
  - Combined effects
- **Display:** "Steep ramp detected at 18:00: +1,450 kW/15min (72% weather, 28% behavioral)"

---

## 4. WEATHER SENSITIVITY KPIs

### KPI 15: Cooling Load Sensitivity
- **Metric:** Load increase per °C above cooling threshold
- **Formula:** `ΔLoad / ΔTemp` for temperatures > 22°C
- **Typical Range:** 10-20 kW/°C per time block
- **Varies By:**
  - Time of day (higher during afternoon/evening)
  - Day type (higher on weekdays)
  - Baseline load level
- **Display:** "Cooling sensitivity: 15.4 kW/°C during peak hours"

### KPI 16: Heating Load Sensitivity
- **Metric:** Load increase per °C below heating threshold
- **Formula:** `ΔLoad / ΔTemp` for temperatures < 15°C
- **Typical Range:** 8-15 kW/°C per time block
- **Varies By:**
  - Time of day (higher during morning/evening)
  - Wind chill effect
- **Display:** "Heating sensitivity: 12.2 kW/°C during morning hours"

### KPI 17: Humidity-Temperature Interaction Effect
- **Metric:** Additional load due to combined high temp + high humidity
- **Formula:** `Interaction Load = Base Temp Impact × (1 + Humidity Modifier)`
- **Typical Modifier:** +5-15% when humidity > 70%
- **Display:** "High humidity amplified cooling load by +720 kW (18%)"

### KPI 18: Precipitation Load Response
- **Metric:** Load reduction during rain events
- **Breakdown:**
  - Light rain (< 5mm): 1-2% reduction
  - Moderate rain (5-15mm): 3-5% reduction
  - Heavy rain (> 15mm): 5-8% reduction
- **Time Dependency:** Greater impact during daytime hours
- **Display:** "Moderate rainfall reduced afternoon load by 3.8%"

---

## 5. DAY TYPE & CALENDAR KPIs

### KPI 19: Weekday vs Weekend Load Differential
- **Metric:** Average load difference between weekdays and weekends
- **Formula:** `(Avg Weekday Load - Avg Weekend Load) / Avg Weekday Load × 100`
- **Typical Range:** 15-25% lower on weekends
- **Time Block Analysis:** Identify which blocks show largest differences
- **Display:** "Weekend load 18.5% lower, primarily in blocks 32-64 (morning/afternoon)"

### KPI 20: Holiday Impact Factor
- **Metric:** Load reduction on holidays vs normal weekdays
- **Formula:** `(Holiday Load - Weekday Baseline) / Weekday Baseline × 100`
- **Categories:**
  - Major holidays: 25-40% reduction
  - Minor holidays: 10-20% reduction
  - Day before/after holiday: 5-15% change
- **Display:** "Independence Day load 32% below weekday baseline"

### KPI 21: Weekend-to-Weekday Transition
- **Metric:** Load change on Monday vs Friday/Sunday
- **Monday Effect:** Often 5-10% higher than mid-week average (catch-up load)
- **Friday Effect:** Often 3-7% lower than mid-week (early departures)
- **Display:** "Monday morning (blocks 20-40) showed +8.2% vs baseline"

### KPI 22: Day-After-Holiday Effect
- **Metric:** Load pattern on return-to-work day after holiday
- **Typical Pattern:** Gradual ramp-up, not immediate return to normal
- **Formula:** `(Day-After Load - Normal Weekday) / Normal Weekday × 100`
- **Display:** "Tuesday after Memorial Day: 92% of normal weekday load"

---

## 6. BASELINE QUALITY KPIs

### KPI 23: Baseline Confidence Score
- **Components:**
  - Sample Size Weight (40%): More days = higher confidence
  - Data Completeness (25%): < 2% missing = 100%, 5% = 80%, >10% = fail
  - Pattern Stability (25%): Lower std dev = higher score
  - Weather Similarity (10%): Similar weather in baseline period
- **Formula:** `Weighted Score (0-100%)`
- **Target:** > 85%
- **Display:** "Baseline confidence: 87% (7 days, 99.2% complete, low variance)"

### KPI 24: Baseline Standard Deviation
- **Metric:** Variability of load across baseline period
- **Formula:** `StdDev of load for each time block across N days`
- **Low StdDev:** Predictable, reliable baseline
- **High StdDev:** Volatile pattern, less reliable
- **Display per block:** "Block 48 baseline: 24,200 kW ± 1,350 kW (5.6% CV)"

### KPI 25: Day-Type Match Quality
- **Metric:** How well does selected day-type match actual pattern
- **Method:** Compare actual load to day-type baseline (weekday/weekend/holiday)
- **Formula:** `1 - (MAPE of actual vs day-type baseline)`
- **Display:** "Day-type match: 94.2% (correctly classified as weekday)"

### KPI 26: Outlier Days in Baseline Period
- **Metric:** Number of days in baseline that were statistical outliers
- **Detection:** Days where load > 2 standard deviations from mean
- **Impact:** Outliers skew baseline, reduce accuracy
- **Action:** Flag and optionally exclude from baseline
- **Display:** "2 of 7 baseline days flagged as outliers (extreme heat events)"

---

## 7. OPERATIONAL DECISION KPIs

### KPI 27: Reserve Margin Adequacy
- **Formula:** `(Available Capacity - Peak Forecast) / Peak Forecast × 100`
- **Target Ranges:**
  - > 20%: Excessive (over-committed, costly)
  - 15-20%: Optimal (adequate buffer)
  - 10-15%: Adequate (monitor closely)
  - 5-10%: Tight (high risk)
  - < 5%: Critical (emergency actions required)
- **Display:** "Reserve margin: 15.2% (4,330 kW buffer) ✅ Optimal"

### KPI 28: Generator Commitment Optimization
- **Metric:** Alignment of committed capacity with actual peak
- **Target:** Actual peak within 95-105% of committed capacity
- **Outcomes:**
  - Perfect: 95-105% (efficient commitment)
  - Over-committed: < 95% (wasted fuel, increased costs)
  - Under-committed: > 105% (emergency reserves, reliability risk)
- **Display:** "Actual peak: 98.3% of committed capacity ✅"

### KPI 29: Demand Response Activation Accuracy
- **Metric:** How often did forecast correctly predict need for DR
- **Categories:**
  - True Positive: Forecast triggered DR, was needed
  - True Negative: Forecast didn't trigger DR, wasn't needed
  - False Positive: Triggered DR unnecessarily (customer impact)
  - False Negative: Didn't trigger DR when needed (reliability risk)
- **Target:** > 90% accuracy (TP + TN)
- **Display:** "DR accuracy: 92% (23 correct, 2 false alarms)"

### KPI 30: Peak Time Prediction Accuracy
- **Metric:** Did forecast correctly identify peak timing?
- **Categories:**
  - Exact match: Same time block
  - Close: Within ±2 blocks (±30 minutes)
  - Moderate error: Within ±4 blocks (±1 hour)
  - Poor: > ±4 blocks
- **Business Impact:** Affects unit commitment timing, DR scheduling
- **Display:** "Peak forecast: 18:00, Actual: 18:15 ✅ Within 1 block"

---

## 8. FINANCIAL & BUSINESS IMPACT KPIs

### KPI 31: Energy Procurement Cost Variance
- **Metric:** Financial impact of forecast error on energy purchases
- **Formula:** `|Actual Energy Cost - Planned Energy Cost based on Forecast|`
- **Components:**
  - Over-forecast: Bought excess energy, may have sold at loss
  - Under-forecast: Emergency purchases at higher prices
- **Display:** "Forecast error cost: $2,340 (under-bought 1,850 kWh at peak price)"

### KPI 32: Capacity Payment Efficiency
- **Metric:** Optimization of capacity commitments
- **Formula:** `Wasted Capacity Cost = (Committed - Actual Peak) × Capacity Price`
- **Target:** Minimize excess capacity payments
- **Display:** "Capacity efficiency: 96.8% (saved $1,200 vs 20% standard margin)"

### KPI 33: Ancillary Services Cost Impact
- **Metric:** Cost of corrective actions due to forecast errors
- **Categories:**
  - Regulation services (minute-to-minute balancing)
  - Spinning reserves (emergency capacity)
  - Non-spinning reserves
- **Display:** "Ancillary services cost: $850 (forecast error required 1,200 kW regulation)"

### KPI 34: Fuel Cost Variance
- **Metric:** Fuel cost difference due to suboptimal unit commitment
- **Impact:** Forecast errors → inefficient generation mix
- **Display:** "Fuel cost variance: +$3,100 (ran expensive peaker for 2 hours)"

---

## 9. BLOCK-LEVEL VARIANCE ATTRIBUTION KPIs

### KPI 35: Variance Attribution by Time Block
**For each of 96 blocks, decompose variance into:**

#### Component A: Weather Attribution
- Temperature contribution: kW and %
- Humidity contribution: kW and %
- Precipitation contribution: kW and %
- Combined weather effect: kW and %

#### Component B: Calendar Attribution
- Day-type effect: kW and %
- Seasonal effect: kW and %
- Holiday proximity: kW and %

#### Component C: Operational Attribution
- Demand response events: kW and %
- Planned outages: kW and %
- Special events: kW and %

#### Component D: Unexplained Variance
- Residual: kW and %
- Target: < 15% unexplained

**Example Block Analysis:**
```
Block 72 (18:00) Variance Analysis
Total Variance: +4,800 kW (+18.2% vs baseline)

Attribution:
├─ Weather Impact: +4,100 kW (85.4%)
│  ├─ Temperature (32°C vs 25°C baseline): +3,400 kW (70.8%)
│  ├─ Humidity (85% vs 60% baseline): +900 kW (18.8%)
│  └─ Precipitation (0mm, baseline 0mm): 0 kW (0%)
│
├─ Calendar Impact: +300 kW (6.3%)
│  └─ Weekday peak pattern: +300 kW (6.3%)
│
├─ Operational Impact: 0 kW (0%)
│
└─ Unexplained: +400 kW (8.3%) ✅ Within target
```

### KPI 36: Peak Load Variance Attribution
- **Metric:** Why did peak load differ from baseline?
- **Full decomposition of peak load variance**
- **Identify dominant factor** (temperature, humidity, day-type, etc.)
- **Display:** "Peak variance: +2,850 kW → 78% temperature, 15% humidity, 7% other"

### KPI 37: Energy Variance Attribution (Daily Total)
- **Metric:** Why did total daily energy differ from baseline?
- **Aggregate 96 blocks into daily attribution**
- **Display:** "Daily energy +18,500 kWh → 62% weather, 25% calendar, 13% unexplained"

---

## 10. TIME-BASED PATTERN KPIs

### KPI 38: Morning Ramp Attribution
- **Period:** Blocks 20-32 (05:00-08:00)
- **Metric:** What drives morning load increase?
- **Typical Factors:**
  - Work start time (behavioral)
  - Temperature rise (weather)
  - HVAC systems startup
- **Display:** "Morning ramp: 65% behavioral, 25% weather, 10% HVAC startup"

### KPI 39: Midday Plateau Attribution
- **Period:** Blocks 32-56 (08:00-14:00)
- **Metric:** What maintains/changes midday load level?
- **Typical Factors:**
  - Cooling load (sunny days)
  - Commercial activity level
  - Cloud cover changes
- **Display:** "Midday load: 70% business activity, 20% cooling, 10% baseline"

### KPI 40: Evening Peak Attribution
- **Period:** Blocks 64-76 (16:00-19:00)
- **Metric:** What drives evening peak?
- **Typical Factors:**
  - Residential return (behavioral)
  - Continued cooling load
  - Lighting and appliances
- **Display:** "Evening peak: 45% residential, 35% cooling, 20% commercial tail"

### KPI 41: Night Valley Attribution
- **Period:** Blocks 88-96, 1-16 (22:00-04:00)
- **Metric:** What determines overnight low?
- **Typical Factors:**
  - Heating load (winter nights)
  - Always-on loads
  - Industrial processes
- **Display:** "Night load: 60% base load, 30% heating, 10% industrial"

---

## 11. MULTI-FACTOR INTERACTION KPIs

### KPI 42: Temperature-Humidity Heat Index Effect
- **Metric:** Combined impact of high temp + high humidity
- **Formula:** `Heat Index = f(Temperature, Humidity)`
- **Load Response:** Non-linear increase above heat index 80°F (27°C)
- **Display:** "Heat index 95°F (+12° vs temp alone) added +1,200 kW cooling load"

### KPI 43: Weekend + Weather Interaction
- **Metric:** How weather impacts change on weekends vs weekdays
- **Observation:** Temperature sensitivity often lower on weekends
- **Reason:** Commercial cooling loads reduced
- **Display:** "Weekend cooling sensitivity: 60% of weekday rate"

### KPI 44: Holiday + Extreme Weather Interaction
- **Metric:** Load pattern during holidays with extreme weather
- **Typical Pattern:** Holiday reduction partially offset by weather load
- **Display:** "Holiday load -30%, but extreme heat added back +12% → net -18%"

### KPI 45: Rain + Temperature Interaction
- **Metric:** Precipitation impact varies by temperature
- **Pattern:**
  - Hot day + rain: Cooling load drops significantly
  - Cold day + rain: Minimal impact or slight heating increase
- **Display:** "Rain on hot day reduced cooling load by 6.8%, vs 1.2% on mild day"

---

## 12. FORECAST IMPROVEMENT KPIs

### KPI 46: Error Reduction Over Time
- **Metric:** Month-over-month MAPE improvement
- **Target:** 5-10% reduction per quarter as model matures
- **Display:** "Q1 MAPE: 3.2% → Q2 MAPE: 2.8% (12.5% improvement) ✅"

### KPI 47: Systematic Error Elimination
- **Metric:** Reduction in forecast bias
- **Track:** Monthly bias trending toward zero
- **Display:** "Bias: Jan +450 kW → Feb +180 kW → Mar +20 kW ✅ Improving"

### KPI 48: Problematic Block Identification Rate
- **Metric:** % of high-error blocks with identified root cause
- **Target:** > 80% of errors explained
- **Action:** Fix identified issues in next version
- **Display:** "18 of 22 high-error blocks (82%) root cause identified ✅"

---

## 13. DATA QUALITY KPIs

### KPI 49: Historical Data Completeness
- **Metric:** % of time blocks with valid load data
- **Target:** > 98% for baseline calculations
- **Impact:** Missing data degrades baseline quality
- **Display:** "Last 30 days: 99.6% complete (10 of 2,880 blocks missing) ✅"

### KPI 50: Weather Data Accuracy
- **Metric:** Match between forecasted weather and actual weather
- **Measurement:** Compare weather forecast used vs actual observed
- **Impact:** Weather forecast errors compound load forecast errors
- **Display:** "Weather forecast MAPE: 1.2°C temp, 8% humidity, 40% precip"

---

## KPI DASHBOARD PRIORITY LEVELS

### **Level 1: Real-Time Operations (Every 15 minutes)**
- Current load vs forecast (KPI tracking)
- Reserve margin (KPI 27)
- Next peak time prediction (KPI 30)
- Weather variance alert (KPIs 7-9)

### **Level 2: Day-Ahead Planning (Daily)**
- Tomorrow's forecast with full attribution (KPIs 11, 35)
- Peak load forecast and attribution (KPI 36)
- Ramp rate warnings (KPI 14)
- Capacity commitment recommendation (KPI 28)

### **Level 3: Performance Review (Weekly)**
- 7-day MAPE and accuracy metrics (KPIs 1-6)
- Variance attribution trends (KPI 37)
- Cost impact summary (KPIs 31-34)
- Baseline quality assessment (KPIs 23-26)

### **Level 4: Strategic Analysis (Monthly)**
- Forecast improvement trends (KPIs 46-48)
- Weather sensitivity recalibration (KPIs 15-18)
- Calendar pattern validation (KPIs 19-22)
- Data quality assessment (KPIs 49-50)

---

## KPI INTERPRETATION GUIDE

### When Load Rises Above Baseline:

**Step 1: Check Time Block**
- Is this typically a peak period? (blocks 64-76)
- If yes: May be normal variance
- If no: Investigate further

**Step 2: Check Weather Attribution (KPIs 7-9)**
- Temperature rise? → Cooling load (KPI 15)
- Temperature drop? → Heating load (KPI 16)
- High humidity? → Amplified cooling (KPI 17)
- No weather change? → Go to Step 3

**Step 3: Check Calendar Attribution (KPIs 19-22)**
- Return from weekend/holiday? (KPI 21, 22)
- Special event day?
- Baseline day-type mismatch? (KPI 25)

**Step 4: Check Unexplained Variance (KPI 35)**
- If < 15%: Well understood ✅
- If > 15%: Requires investigation ⚠️

### When Load Dips Below Baseline:

**Step 1: Check Precipitation (KPI 9)**
- Rain reduces load 2-5% during daytime

**Step 2: Check Calendar (KPIs 19-22)**
- Holiday, weekend, or day-before-holiday effect

**Step 3: Check Temperature (KPIs 7-8)**
- Mild temperatures (15-22°C) reduce HVAC load

**Step 4: Check for Operational Events**
- Demand response activation
- Planned outage
- Special circumstances

---

This comprehensive KPI framework enables complete attribution of every load variation to specific root causes, supporting both operational decision-making and continuous forecast improvement.