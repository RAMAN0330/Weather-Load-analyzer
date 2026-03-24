    # Frontend Design Guide - Load Simulation Application
## Complete UI/UX Design Plan

---

## 1. DESIGN PHILOSOPHY

### Core Principles
- **Operations-First:** Grid operators need answers fast (< 5 seconds to insight)
- **Attribution-Driven:** Every variance must show "why" immediately
- **Actionable Intelligence:** Don't just show data, recommend actions
- **Progressive Disclosure:** Simple overview → detailed analysis on-demand
- **Real-Time Awareness:** Live updates without overwhelming the user

### Design Goals
- **Decision Speed:** From question to action in < 30 seconds
- **Cognitive Load:** No more than 3 clicks to any critical information
- **Visual Hierarchy:** Most critical KPIs always visible
- **Error Prevention:** Validate inputs, confirm critical actions
- **Mobile-Ready:** Key metrics accessible on tablets/phones

---

## 2. INFORMATION ARCHITECTURE

### Primary Navigation Structure

```
┌─────────────────────────────────────────────────────────┐
│  LOGO   [Day-Ahead] [Live Ops] [Analysis] [History] ⚙️ │
└─────────────────────────────────────────────────────────┘
```

**Top-Level Pages:**

1. **Day-Ahead Forecast** (Default landing page)
   - Tomorrow's 96-block forecast
   - Weather scenario planning
   - Quick attribution insights

2. **Live Operations Dashboard**
   - Real-time load tracking
   - Current vs forecast comparison
   - Active alerts and actions

3. **Attribution Analysis**
   - Deep-dive variance decomposition
   - Multi-factor correlation analysis
   - Pattern discovery

4. **Historical Performance**
   - Forecast accuracy trends
   - Cost impact reports
   - Model improvement tracking

5. **Settings & Configuration**
   - Weather sensitivity rules
   - Baseline preferences
   - Alert thresholds
   - User preferences

---

## 3. PAGE-BY-PAGE DESIGN

---

## PAGE 1: DAY-AHEAD FORECAST DASHBOARD

### Layout Structure (Desktop: 1920x1080)

```
┌──────────────────────────────────────────────────────────────────┐
│  Header: Quick Controls + Key Metrics (100px height)            │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Main Chart: 96-Block Forecast (500px height)                   │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│  Left: Weather Config      │  Right: Attribution Breakdown       │
│  (350px width)            │  (remaining width)                  │
│  (400px height)           │                                     │
└──────────────────────────────────────────────────────────────────┘
```

---

### Section A: Header Controls & KPI Cards

**Layout:** Single row, 5 columns

```
┌────────────┬────────────┬────────────┬────────────┬────────────┐
│ Controls   │  Peak Load │Daily Energy│  Reserve   │ Confidence │
│            │            │            │  Margin    │   Score    │
└────────────┴────────────┴────────────┴────────────┴────────────┘
```

#### Column 1: Forecast Controls (280px wide)

**Visual Design:**
```
┌─────────────────────────────────┐
│ Date: [2026-02-12 ▼]           │
│ Type: Weekday ⚡ Auto-detected │
│ Baseline: [7 days ◀━━━━○━━▶ 15]│
│                                 │
│ [🔄 Generate Forecast]         │
└─────────────────────────────────┘
```

**Elements:**
- **Date Picker:** Calendar dropdown, default = tomorrow
- **Day Type Indicator:** Badge with icon (📅 weekday, 🏖️ weekend, 🎉 holiday)
- **Baseline Slider:** Visual slider with number labels
- **Primary Button:** Large, prominent "Generate Forecast" button
  - Color: Primary blue (#2563eb)
  - Size: Full width, 44px height
  - Icon: Refresh/play icon

#### Columns 2-5: Live KPI Cards (Each ~300px wide)

**Card Design Pattern:**
```
┌─────────────────────────────────┐
│ PEAK LOAD FORECAST              │
│ 28,450 kW                       │ ← Large, bold number
│ @ 18:00 (Block 72)             │ ← Context
│ ─────────────                   │
│ Baseline: 27,120 kW            │ ← Comparison
│ +1,330 kW (+4.9%) ⬆️           │ ← Delta with icon
│                                 │
│ [View Details →]               │ ← Action link
└─────────────────────────────────┘
```

**Card 1: Peak Load**
- **Primary Metric:** Peak kW (48px font, bold)
- **Context:** Time and block number (18px font)
- **Comparison:** Baseline value and delta
- **Status Indicator:** 
  - Green ✅ if within ±3%
  - Yellow ⚠️ if ±3-7%
  - Red 🚨 if >±7%

**Card 2: Daily Energy**
- **Primary Metric:** Total kWh (48px font)
- **Comparison:** Baseline total and % deviation
- **Cost Impact:** Estimated $ variance (if error exists)

**Card 3: Reserve Margin**
- **Primary Metric:** Percentage (48px font)
- **Visual:** Mini progress bar
  - Green: >15%
  - Yellow: 10-15%
  - Red: <10%
- **Secondary:** Absolute kW buffer

**Card 4: Confidence Score**
- **Primary Metric:** Percentage (48px font)
- **Breakdown:** Hoverable tooltip showing:
  - Baseline quality: 92%
  - Weather certainty: 85%
  - Day-type match: 94%
- **Visual:** Circular progress indicator

---

### Section B: Main 96-Block Forecast Chart

**Chart Dimensions:** Full width × 500px height

#### Chart Type: Multi-Layer Area + Line Chart

**Visual Layers (Bottom to Top):**

1. **Background Layer:** Shaded regions
   - Light blue: Off-peak hours (blocks 1-32, 85-96)
   - Light yellow: Peak hours (blocks 60-80)
   - Light red: User-modified weather blocks

2. **Baseline Range:** Gray shaded band
   - Upper bound: Baseline max (from lookback period)
   - Lower bound: Baseline min
   - Opacity: 20%

3. **Baseline Average Line:**
   - Color: Gray (#6b7280)
   - Style: Dashed (4px dash, 4px gap)
   - Width: 2px

4. **Forecasted Load Line:**
   - Color: Primary blue (#2563eb)
   - Style: Solid
   - Width: 3px
   - Highlight: Glowing effect on hover

5. **Historical Actual Line** (if validating past forecast)
   - Color: Green (#10b981)
   - Style: Solid
   - Width: 2px

6. **Data Points:** Circles at each block
   - Size: 6px diameter
   - Visible on hover only (to reduce clutter)

#### X-Axis Design

```
00:00    04:00    08:00    12:00    16:00    20:00    24:00
  │        │        │        │        │        │        │
  └────────┴────────┴────────┴────────┴────────┴────────┘
        Time Block: 1  2  3  4 ... 94 95 96
```

**Configuration:**
- **Primary Labels:** Hour markers (00:00, 04:00, etc.) every 16 blocks
- **Secondary Labels:** Time block numbers (smaller, gray)
- **Grid Lines:** Vertical lines every 4 blocks (1-hour intervals)
  - Major lines (4-hour): 1px, #e5e7eb
  - Minor lines (1-hour): 0.5px, #f3f4f6

#### Y-Axis Design

**Configuration:**
- **Range:** Auto-scale to min/max with 10% padding
- **Labels:** kW or MW (auto-convert if >10,000 kW)
- **Grid Lines:** Horizontal, every 2,000 kW
- **Format:** Comma separators (e.g., "24,500 kW")

#### Interactive Features

**Hover Tooltip:**
```
┌─────────────────────────────────┐
│ Block 72 • 18:00               │
│                                 │
│ Forecasted: 28,450 kW          │
│ Baseline:   27,120 kW          │
│ Deviation:  +1,330 kW (+4.9%)  │
│ ─────────────────────────────── │
│ Temperature:  32°C (+7°C)      │
│ Humidity:     85% (+25%)       │
│ Precipitation: 0mm             │
│ ─────────────────────────────── │
│ Attribution:                    │
│ • Temperature: +1,100 kW (83%) │
│ • Humidity:      +230 kW (17%) │
│                                 │
│ [Edit Weather →]               │
└─────────────────────────────────┘
```

**Click Interaction:**
- **Single Block Click:** Opens weather editor for that block
- **Drag Selection:** Select range of blocks (e.g., 60-72)
  - Visual: Semi-transparent overlay on selected region
  - Action: "Edit 13 blocks" button appears

**Zoom & Pan:**
- **Scroll Wheel:** Zoom in/out on X-axis
- **Click + Drag:** Pan left/right when zoomed
- **Reset Button:** Return to full 96-block view

#### Chart Annotations

**Peak Marker:**
```
        ⬆️ PEAK
       28,450 kW
         18:00
```
- Visual: Arrow pointing to peak point
- Color: Red (#ef4444)
- Always visible

**Significant Variance Markers:**
- Blocks with >±5% deviation get warning icon ⚠️
- Blocks with >±10% deviation get alert icon 🚨

---

### Section C: Weather Configuration Panel (Left Side)

**Dimensions:** 350px wide × 400px height

**Panel Design:**
```
┌─────────────────────────────────────────┐
│ WEATHER CONFIGURATION                    │
│ ─────────────────────────────────────── │
│                                          │
│ [All Blocks] [Range: 60-72] [Single: -] │ ← Tab selector
│                                          │
│ Temperature                              │
│ [━━━━━○━━━━━━━━] 25°C                    │
│ Range: 15°C ────────── 35°C             │
│                                          │
│ Humidity                                 │
│ [━━━━━━○━━━━━━] 60%                      │
│ Range: 30% ────────── 90%               │
│                                          │
│ Precipitation                            │
│ [━━━━○━━━━━━━━] 0 mm                     │
│ Range: 0 mm ────────── 25 mm            │
│                                          │
│ ─────────────────────────────────────── │
│                                          │
│ Quick Presets:                           │
│ [☀️ Hot Day] [❄️ Cold] [🌧️ Rainy]       │
│                                          │
│ [Apply Changes]  [Reset to Baseline]    │
└─────────────────────────────────────────┘
```

#### Tab Modes

**Mode 1: All Blocks**
- Apply same weather to all 96 blocks
- Use case: Testing uniform condition change

**Mode 2: Range Selection**
- Input: "From block 60 to block 72"
- Apply weather to selected range
- Visual: Selected range highlighted in chart

**Mode 3: Single Block**
- Dropdown: Select specific block
- Fine-tuned control for critical periods

#### Slider Design

**Temperature Slider:**
- **Visual:** Blue-to-red gradient track
- **Handle:** Circular, 24px diameter
- **Current Value:** Large number next to slider
- **Real-Time Update:** Chart updates as you drag
- **Input Field:** Can type exact value

**Humidity Slider:**
- **Visual:** Blue gradient (light to dark)
- **Range:** 0-100%
- **Step:** 5% increments

**Precipitation Slider:**
- **Visual:** Gray to dark blue
- **Range:** 0-50mm
- **Step:** 0.5mm increments

#### Quick Preset Buttons

**Visual Design:**
```
┌──────────────┐
│   ☀️ Hot Day  │ ← Icon + label
├──────────────┤
│ Temp: 35°C   │ ← Preview values
│ Humid: 75%   │
│ Rain: 0mm    │
└──────────────┘
```

**Presets:**
1. **Hot Day:** 35°C, 75%, 0mm
2. **Cold Day:** 5°C, 60%, 0mm
3. **Rainy Day:** 18°C, 90%, 15mm
4. **Mild Day:** 22°C, 50%, 0mm

**Custom Presets:** User can save their own

#### Action Buttons

**Primary Button: "Apply Changes"**
- Color: Blue (#2563eb)
- Effect: Triggers forecast recalculation
- Disabled if no changes made

**Secondary Button: "Reset to Baseline"**
- Color: Gray outline
- Effect: Reverts to historical weather data

---

### Section D: Attribution Breakdown Panel (Right Side)

**Dimensions:** Remaining width × 400px height

**Panel Design:**
```
┌─────────────────────────────────────────────────────────┐
│ VARIANCE ATTRIBUTION                                     │
│ ─────────────────────────────────────────────────────── │
│                                                          │
│ Total Variance: +4,280 kWh (+1.8%) from baseline       │
│                                                          │
│ ┌───────────────────────────────────────────────────┐  │
│ │ Temperature Impact        +3,200 kWh  74.8% ████ │  │
│ │ Humidity Impact             +720 kWh  16.8% ██   │  │
│ │ Calendar Effect             +200 kWh   4.7% ▌    │  │
│ │ Precipitation                  0 kWh   0.0%      │  │
│ │ Unexplained                 +160 kWh   3.7% ▌    │  │
│ └───────────────────────────────────────────────────┘  │
│                                                          │
│ 🔍 Insight:                                             │
│ "High temperatures (avg 28°C, +6°C above baseline)     │
│  driving 75% of load increase. Peak impact during      │
│  blocks 60-76 (afternoon/evening cooling load)."       │
│                                                          │
│ ⚡ Recommended Actions:                                 │
│ • Pre-position 2,500 kW peaking capacity for 17:00    │
│ • Consider demand response for blocks 68-76            │
│ • Monitor cooling load during afternoon                │
│                                                          │
│ [View Detailed Analysis →]                             │
└─────────────────────────────────────────────────────────┘
```

#### Visual Elements

**Horizontal Bar Chart:**
- Each factor = one bar
- Bar length = percentage contribution
- Color coding:
  - Temperature: Red gradient
  - Humidity: Blue
  - Precipitation: Dark blue
  - Calendar: Purple
  - Unexplained: Gray

**Insight Box:**
- **Icon:** 🔍 magnifying glass
- **Background:** Light blue (#eff6ff)
- **Content:** Plain-language explanation
- **Auto-generated:** Based on dominant factor

**Action Recommendations:**
- **Icon:** ⚡ lightning bolt
- **Format:** Bulleted list
- **Content:** Specific operational actions
- **Conditional:** Only appears if variance >±3%

---

## PAGE 2: LIVE OPERATIONS DASHBOARD

### Layout Structure

```
┌──────────────────────────────────────────────────────────────┐
│  Current Status Bar (80px height)                           │
├──────────────────────────────────────────────────────────────┤
│  Live Chart: Today's Progress (400px)                        │
├──────────────────────────────────────────────────────────────┤
│  Left: Active Alerts     │  Right: Next 6 Hours Forecast     │
│  (50% width)            │  (50% width)                      │
└──────────────────────────────────────────────────────────────┘
```

### Current Status Bar

```
┌────────────┬────────────┬────────────┬────────────┬────────────┐
│ Current    │ Forecast   │ Accuracy   │  Reserve   │   Time to  │
│  Load      │ for Now    │  Today     │  Margin    │    Peak    │
│            │            │            │            │            │
│ 24,820 kW  │ 24,650 kW  │  MAPE:    │   15.2%    │  3h 28min  │
│            │ (-0.7%) ✅ │   1.8% ✅  │  ✅ Good   │  @ 18:00   │
└────────────┴────────────┴────────────┴────────────┴────────────┘
```

**Auto-Update:** Every 15 minutes (live WebSocket connection)

### Live Progress Chart

**Visual Design:**
- **X-Axis:** Current day's 96 blocks
- **Current Time Marker:** Vertical red line with "NOW" label
- **Past Blocks:** 
  - Forecasted: Blue line
  - Actual: Green line
  - Error band: Red/green shaded area
- **Future Blocks:** 
  - Forecasted: Blue dashed line
  - Confidence band: Light blue shaded area

**Annotations:**
```
    NOW ↓
    14:30
    
Past ←  |  → Future
(Actual)| (Forecast)
```

### Active Alerts Panel

```
┌─────────────────────────────────────────┐
│ 🚨 ACTIVE ALERTS (2)                    │
│ ─────────────────────────────────────── │
│                                          │
│ ⚠️  HIGH VARIANCE - Block 56            │
│     14:00 | Actual 6.2% above forecast  │
│     Cause: Temperature +3°C unexpected  │
│     [View Details] [Dismiss]            │
│                                          │
│ ℹ️  PEAK APPROACHING - 3h 28min         │
│     Expected: 28,450 kW @ 18:00         │
│     Tracking: On target (±2%)           │
│     [Pre-Position Units]                │
│                                          │
│ ✅ No critical issues                   │
└─────────────────────────────────────────┘
```

**Alert Priority:**
1. 🚨 Critical (red): >10% variance, reserve <10%
2. ⚠️ Warning (yellow): 5-10% variance, reserve 10-15%
3. ℹ️ Info (blue): FYI, no action needed

---

## PAGE 3: ATTRIBUTION ANALYSIS

### Layout: Deep-Dive Interface

```
┌──────────────────────────────────────────────────────────────┐
│  Analysis Controls (Date, Block Range, Filters)             │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Waterfall Chart: Variance Decomposition                     │
│  (Shows: Baseline → Temperature → Humidity → etc → Final)   │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│  Left: Factor Details    │  Right: Correlation Matrix       │
└──────────────────────────────────────────────────────────────┘
```

### Waterfall Chart

**Visual Concept:**
```
Baseline    +Temp    +Humid   +Calendar  =Final
25,000 ──┐           
         │ +3,200                        
         ├─────┐                         
         │     │ +720                    
         │     ├────┐                    
         │     │    │ +200               
         │     │    ├───┐                
         │     │     │    │              
    ─────┴─────┴─────┴────┴────── = 29,120 kW
```

**Interpretation:**
- Start: Baseline load
- Each step: Attribution factor
- End: Final forecasted/actual load

### Factor Detail Cards

**Temperature Card:**
```
┌─────────────────────────────────────┐
│ TEMPERATURE IMPACT                  │
│ ─────────────────────────────────── │
│                                      │
│ Contribution: +3,200 kW (74.8%)     │
│                                      │
│ Details:                             │
│ • Actual: 28°C                      │
│ • Baseline: 22°C                    │
│ • Deviation: +6°C                   │
│ • Sensitivity: 533 kW/°C            │
│                                      │
│ Time Pattern:                        │
│ [Mini chart showing temp vs load]   │
│                                      │
│ Peak Impact: Blocks 68-76 (17-19h) │
└─────────────────────────────────────┘
```

---

## PAGE 4: HISTORICAL PERFORMANCE

### Layout: Analytics Dashboard

```
┌──────────────────────────────────────────────────────────────┐
│  Time Period Selector + KPI Filters                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Accuracy Trends: MAPE over Time (Line Chart)               │
│                                                              │
├────────────────────────────┬─────────────────────────────────┤
│  Error Distribution        │  Cost Impact Summary            │
│  (Histogram)              │  (Table + Total)                │
└────────────────────────────┴─────────────────────────────────┘
```

### MAPE Trend Chart

**Design:**
```
│
5%├─────────────────────────────────────
  │                    ●
4%├                ●       ●
  │            ●               ●
3%├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  ← Target
  │    ●   ●                       ●
2%├●                                   ●
  │
1%├
  │
  └─────────────────────────────────────
   Jan  Feb  Mar  Apr  May  Jun  Jul
```

**Elements:**
- **Target Line:** Dashed horizontal line at 3%
- **Data Points:** Connected line with dots
- **Trend Arrow:** ↗️ improving or ↘️ degrading
- **Stats:** Min, max, average displayed

---

## 4. COMPONENT LIBRARY

### Reusable UI Components

#### Component: KPI Card

**Usage:** Display key metrics with context

**Visual Variants:**

**Standard KPI Card:**
```
┌─────────────────────────┐
│ METRIC NAME             │
│                         │
│ 28,450 kW              │ ← Large primary value
│                         │
│ Baseline: 27,120       │ ← Comparison
│ +1,330 (+4.9%) ⬆️      │ ← Delta with trend
│                         │
│ Status: ✅ Normal       │ ← Status indicator
└─────────────────────────┘
```

**Compact KPI Card:**
```
┌──────────────┐
│ Peak Load    │
│ 28,450 kW ⬆️ │
│ +4.9%        │
└──────────────┘
```

**Props:**
- `title`: String
- `value`: Number
- `unit`: String (kW, kWh, %, etc.)
- `baseline`: Number (optional)
- `status`: 'good' | 'warning' | 'critical'
- `trend`: 'up' | 'down' | 'neutral'
- `onClick`: Function (optional, for drill-down)

#### Component: Attribution Bar

**Usage:** Show contribution breakdown

**Visual:**
```
Temperature    ████████████████████  80%  +3,200 kW
Humidity       ████                  15%    +600 kW
Calendar       ██                     5%    +200 kW
```

**Props:**
- `factors`: Array of {name, value, percentage, color}
- `total`: Number
- `showValues`: Boolean
- `showPercentages`: Boolean

#### Component: Time Block Selector

**Usage:** Select specific blocks or ranges

**Visual:**
```
┌─────────────────────────────────────────────┐
│ [All] [Peak Hours] [Custom Range]          │
│                                             │
│ From: [Block 60 ▼] 15:00                   │
│ To:   [Block 72 ▼] 18:00                   │
│                                             │
│ Selected: 13 blocks                         │
└─────────────────────────────────────────────┘
```

**Quick Presets:**
- All (1-96)
- Peak Hours (60-80)
- Morning Ramp (20-32)
- Evening Peak (64-76)
- Night Valley (88-16)

#### Component: Weather Input Group

**Usage:** Grouped weather parameter inputs

**Visual:**
```
┌─────────────────────────────────────────┐
│ Temperature    [━━━━○━━━━] 25°C         │
│                Min: 15°C  Max: 35°C     │
│                                          │
│ Humidity       [━━━○━━━━] 60%           │
│                Min: 30%   Max: 90%      │
│                                          │
│ Precipitation  [━○━━━━━━] 2 mm          │
│                Min: 0mm   Max: 25mm     │
└─────────────────────────────────────────┘
```

#### Component: Alert Badge

**Usage:** Show status and alerts

**Variants:**

**Critical:**
```
┌─────────────────────────┐
│ 🚨 HIGH VARIANCE        │
│ Actual +8.2% forecast   │
└─────────────────────────┘
```

**Warning:**
```
┌─────────────────────────┐
│ ⚠️  RESERVE LOW         │
│ 12.3% (Target: 15%)     │
└─────────────────────────┘
```

**Success:**
```
┌─────────────────────────┐
│ ✅ ON TARGET            │
│ Forecast within ±2%     │
└─────────────────────────┘
```

---

## 5. COLOR SYSTEM

### Primary Palette

**Brand Colors:**
- **Primary Blue:** `#2563eb` - Actions, links, forecasted line
- **Primary Dark:** `#1e40af` - Hover states
- **Primary Light:** `#dbeafe` - Backgrounds

**Semantic Colors:**
- **Success Green:** `#10b981` - Good status, actual line, within target
- **Warning Yellow:** `#f59e0b` - Caution, moderate deviation
- **Error Red:** `#ef4444` - Critical alerts, high variance
- **Info Blue:** `#3b82f6` - Informational messages

### Data Visualization Palette

**Weather Factors:**
- Temperature: `#ef4444` (red) → `#fb923c` (orange)
- Humidity: `#3b82f6` (blue)
- Precipitation: `#1e3a8a` (dark blue)
- Calendar: `#8b5cf6` (purple)
- Unexplained: `#6b7280` (gray)

**Gradients:**
- Temperature scale: Blue (#3b82f6) → Yellow (#fbbf24) → Red (#ef4444)
- Humidity scale: Light blue (#dbeafe) → Dark blue (#1e3a8a)

### Status Indicators

**Traffic Light System:**
- **Green Zone:** Metrics within target (0-3% error)
- **Yellow Zone:** Moderate concern (3-7% error)
- **Red Zone:** Critical attention needed (>7% error)

---

## 6. TYPOGRAPHY

### Font Stack
```css
font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 
             sans-serif;
```

### Type Scale

**Display (Metrics):**
- **Hero Number:** 48px, font-weight 700 (KPI primary values)
- **Large Number:** 32px, font-weight 600 (Section headers)

**Headings:**
- **H1:** 24px, font-weight 600 (Page titles)
- **H2:** 20px, font-weight 600 (Section titles)
- **H3:** 18px, font-weight 600 (Card titles)

**Body:**
- **Body Large:** 16px, font-weight 400 (Main content)
- **Body:** 14px, font-weight 400 (Standard text)
- **Small:** 12px, font-weight 400 (Labels, captions)

**Special:**
- **Monospace:** 'JetBrains Mono' for time blocks, precise numbers

---

## 7. RESPONSIVE DESIGN

### Breakpoints

```css
/* Mobile */
@media (max-width: 768px)

/* Tablet */
@media (min-width: 769px) and (max-width: 1024px)

/* Desktop */
@media (min-width: 1025px)
```

### Mobile Layout (768px and below)

**Day-Ahead Forecast - Mobile:**

```
┌─────────────────────┐
│ ☰  Day-Ahead       │ ← Hamburger menu
├─────────────────────┤
│ Date: Tomorrow ▼    │
│ [Generate]          │
├─────────────────────┤
│ Peak: 28,450 kW     │
│ +4.9% ⬆️            │
├─────────────────────┤
│                     │
│ [Chart - vertical]  │
│                     │
├─────────────────────┤
│ [Tap for Weather]   │
├─────────────────────┤
│ Attribution:        │
│ Temp: 75% █████     │
│ Other: 25% ██       │
└─────────────────────┘
```

**Key Changes:**
- Hamburger navigation
- Single column layout
- Condensed KPI cards (2×2 grid instead of 4×1)
- Chart rotates 90° or uses horizontal scroll
- Collapsible panels for weather/attribution
- Touch-optimized controls (larger tap targets)

### Tablet Layout (769-1024px)

**Hybrid approach:**
- 2-column layout for KPI cards
- Full-width chart
- Side-by-side weather and attribution (stacked on smaller tablets)

---

## 8. INTERACTION PATTERNS

### Hover States

**Interactive Elements:**
- **Chart Lines:** Glow effect, increase width by 1px
- **Buttons:** Darken background by 10%,
raise shadow
- **Cards:** Subtle lift (box-shadow increase)
- **Data Points:** Show tooltip, highlight connected elements

### Loading States

**Skeleton Screens:**
```
┌─────────────────────┐
│ ░░░░░░░░░░░░░░░     │ ← Shimmer animation
│ ░░░░░ ░░░░░░        │
│                     │
│ ░░░░░░░░░░░░░░░░░░  │
└─────────────────────┘
```

**Progress Indicators:**
- **Forecast Generation:** Linear progress bar with % complete
- **Data Loading:** Spinner with "Loading weather data..."
- **Background Updates:** Toast notification "Data refreshed"

### Error States

**Graceful Degradation:**
```
┌─────────────────────────────┐
│ ⚠️  Unable to load forecast │
│                             │
│ Weather API unavailable     │
│                             │
│ [Retry] [Use Cached Data]  │
└─────────────────────────────┘
```

**Inline Validation:**
- Temperature out of range: Red border + error message
- Date in past: Warning "Cannot forecast past dates"
- Missing data: Info icon with explanation

---

## 9. ACCESSIBILITY

### WCAG 2.1 AA Compliance

**Color Contrast:**
- Text on backgrounds: Minimum 4.5:1 ratio
- Large text (18px+): Minimum 3:1 ratio
- Critical status indicators: Not rely on color alone (use icons)

**Keyboard Navigation:**
- All interactive elements: Tab-accessible
- Focus indicators: Visible outline (2px blue ring)
- Shortcuts:
  - `G`: Generate forecast
  - `W`: Open weather config
  - `Arrow keys`: Navigate time blocks
  - `Esc`: Close modals/panels

**Screen Reader Support:**
- ARIA labels on all charts
- Alt text on icons
- Semantic HTML (header, nav, main, section)
- Live regions for dynamic updates

**Motion:**
- Reduce motion option: Disables animations
- No auto-playing animations (user-triggered only)

---

## 10. PERFORMANCE OPTIMIZATION

### Front-End Performance

**Chart Rendering:**
- Virtual scrolling for 96 blocks
- Canvas-based rendering for large datasets
- Debounce slider updates (300ms)
- Lazy load historical data

**Data Management:**
- Cache forecast results (24h TTL)
- Prefetch tomorrow's weather forecast
- Optimistic UI updates
- Background sync for live data

**Bundle Size:**
- Code splitting by page
- Lazy load attribution analysis
- Tree-shake unused components
- Compress images and assets

### Target Metrics

- **First Contentful Paint:** < 1.5s
- **Time to Interactive:** < 3s
- **Chart Render:** < 500ms
- **Forecast Generation:** < 2s
- **Live Update Latency:** < 1s

---

## 11. DESIGN WORKFLOW

### Design → Development Pipeline

**Step 1: Wireframes** (Week 1)
- Low-fidelity layouts
- Information hierarchy
- User flow diagrams

**Step 2: High-Fidelity Mockups** (Week 2)
- Figma designs
- Interactive prototypes
- Component library

**Step 3: Design System** (Week 3)
- Documented components
- Style guide
- Code snippets

**Step 4: Development Handoff** (Week 4)
- Figma → Code (auto-generate CSS)
- Component specifications
- Interaction documentation

### Tools

**Design:** Figma, Sketch, or Adobe XD
**Prototyping:** Figma, ProtoPie
**Collaboration:** Zeplin, InVision
**Icons:** Lucide React, Heroicons
**Charts:** Recharts, D3.js, Chart.js

---

## 12. SAMPLE UI FLOWS

### Flow 1: First-Time User

1. **Landing:** Day-Ahead Forecast page with sample data
2. **Tooltip Tour:** Highlights key features (auto-plays once)
3. **Quick Action:** "Generate Tomorrow's Forecast" CTA
4. **Success:** Results displayed with explanation tooltips
5. **Next Step:** Prompt to adjust weather or save scenario

### Flow 2: Daily Operations User

1. **Landing:** Live Operations Dashboard (if current day in progress)
2. **At-a-Glance:** See current vs forecast, reserve margin
3. **Deep Dive:** Click variance alert → Attribution analysis
4. **Action:** Review recommendations, dismiss or act
5. **Return:** Back to live dashboard

### Flow 3: Analysis User

1. **Navigate:** Historical Performance page
2. **Select Period:** Last 30 days
3. **Review Trends:** MAPE chart, error distribution
4. **Investigate:** Click high-error day → Drill-down
5. **Insight:** Identify systematic error pattern
6. **Action:** Adjust weather sensitivity rules

---

## 13. ANIMATION & MICRO-INTERACTIONS

### Subtle Animations

**Chart Animations:**
- **On Load:** Lines draw from left to right (800ms ease-out)
- **On Update:** Smooth transitions (400ms)
- **On Hover:** Gentle pulse on data point (200ms)

**Card Animations:**
- **On Load:** Fade up + slight rise (600ms stagger)
- **On Hover:** Lift shadow (200ms ease-out)
- **On Click:** Brief scale (150ms)

**Button Animations:**
- **On Hover:** Background color transition (200ms)
- **On Click:** Scale down (100ms) → scale up (100ms)
- **Loading:** Spinner rotation (continuous)

### Notification Toasts

**Position:** Top-right corner

**Types:**
```
Success: ✅ "Forecast generated successfully"
Warning: ⚠️  "Weather data partially unavailable"
Error:   🚨 "Failed to load baseline data"
Info:    ℹ️  "New weather forecast available"
```

**Behavior:**
- Auto-dismiss: 4 seconds
- Manual dismiss: X button
- Stack: Max 3 toasts visible
- Animation: Slide in from right

---

This comprehensive frontend design guide provides everything needed to build an intuitive, powerful, and operationally-focused load simulation application that enables grid operators to quickly understand load variations and their root causes.