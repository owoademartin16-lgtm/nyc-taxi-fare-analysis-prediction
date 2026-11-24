# NYC Taxi Fare Intelligence

A Streamlit dashboard for New York's yellow-taxi January 2025. It turns the
official TLC trip file into a dispatch-board view of demand, fares and zones,
then puts the saved fare model to work in a live estimator.

The dashboard is built around one idea: a taxi meter is a clock, a map and a
price at the same time. The pages move through those three lenses, then close
with a single briefing that keeps the strongest signals in one place.

## What's inside

| Page | What it does |
|---|---|
| **Overview** | The month in one screen: 3.33M valid trips, $90.3M in meter revenue, the 6 PM peak, the busiest day, weekday rhythm and the top pickup zones. |
| **Demand & Time** | Day × hour heatmap, weekday/weekend rhythms, shift mix, peak windows and the 5 AM average-fare premium. |
| **Fares & Zones** | Fare-vs-distance density, fare per mile by hour, a pickup-demand zone map, top zones, airport rate codes and tip behaviour. |
| **Model Lab** | Holdout model card, actual-vs-predicted density, residuals, model comparison, feature importance and error by fare and distance band. |
| **Fare Estimator** | A live total-amount estimate from the saved LightGBM model, with a typical range and a comparison group of similar real trips. |
| **Insights** | A single briefing that pulls the strongest demand, fare, model and estimator signals into one readable section. |

## The numbers

| Metric | Value |
|---|---:|
| January trips in file | **3,475,204** |
| Negative fare/total rows removed | **144,439** |
| Valid January trips | **3,330,765** |
| Meter revenue | **$90,335,708.80** |
| Average total amount | **$27.12** |
| Median total amount | **$20.35** |
| Median trip distance | **1.67 mi** |
| Solo riders | **82.05%** |
| Peak hour | **6 PM · 253,993 trips** |
| Busiest day | **Jan 16 · 134,129 trips · $3.64M** |
| Weekday trips | **2,490,134** |
| Weekend trips | **840,631** |
| JFK + Newark | **98,694 trips · 2.96% of trips · 10.36% of revenue** |
| Card tip rate | **14.6%** of card totals |

The January file contains **3,475,204** trips. Removing **144,439** rows with a
negative fare or negative total leaves **3,330,765** valid trips, which is the
count used for the exploratory charts.

The fare model is evaluated on the same 80/20 split as the case study
(`random_state=42`) using the **3,229,937** model-eligible trips:

| Holdout metric | Value |
|---|---:|
| MAE | **$2.6147** |
| RMSE | **$4.5708** |
| R² | **0.9484** |
| Train R² | **0.9504** |
| Train–test gap | **0.0020** |
| Holdout rows | **645,988** |
| Errors within $5 | **87.9%** |

## Design

The visual system is a taxi receipt crossed with a dispatch board:

- Warm receipt-paper background (`#F3EADB`) with asphalt hero sections.
- Taxi yellow (`#F5C518`), fare orange (`#D95D39`) and meter teal (`#237A6B`).
- A checkered strip at the bottom of every hero and meter card.
- Bebas Neue for display, Manrope for body copy, IBM Plex Mono for labels.
- No chart gridlines, upper-right legends, clear titles and paired charts at
  equal height.
- Precomputed aggregates and a 350k-trip sample keep the app fast. Nothing is
  trained or downloaded at runtime.

## Run locally

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

The model artifacts are included in `models/`:

- `taxi_model.joblib` — saved LightGBM model
- `taxi_preprocessor.joblib` — fitted `ColumnTransformer`
- `taxi_fallback.joblib` — portable scikit-learn fallback

### macOS and `libomp`

LightGBM needs OpenMP. If `import lightgbm` fails with a missing
`libomp.dylib`, install it with Homebrew:

```bash
brew install libomp
```

The app catches a LightGBM load failure and uses the portable fallback so the
rest of the dashboard still runs. The result card says which model is active.
On Streamlit Community Cloud (Linux), the primary LightGBM model is used.

The preprocessor was fitted with scikit-learn 1.5.2. The loader repairs a
private `_RemainderColsList` class that newer scikit-learn releases removed, so
the saved transformer still loads and produces the model's expected 21 columns.

## Rebuild the assets

```bash
python3 build_assets.py
```

The script downloads the official TLC January 2025 parquet and zone lookup into
`raw/`, applies the cleaning recipe, writes the compact CSVs/parquet files in
`data/`, reproduces the holdout metrics and retrains the fallback. The raw file
is intentionally ignored by Git.

## Data and scope

- Source: [NYC TLC Yellow Taxi Trip Data, January 2025](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
- Zone geometry: TLC taxi zone shapefile, simplified into `data/zone_map.geojson`.
- The overview and demand pages use the full valid January set after removing
  negative fares and invalid totals.
- Fare-shape and model charts use the model-eligible set: distance 0–100 mi,
  total up to $300, and 1–6 passengers.
- The target is `total_amount`, which includes tips, tolls and fees. Cash tips
  are under-reported by nature.
- The fare estimator assumes a January-2025-style trip. It does not know about
  live traffic, weather or route choice.

## Project structure

```text
nyc-taxi-dashboard/
├── app.py                  # Overview entry point + navigation
├── pages/
│   ├── 1_Demand_and_Time.py
│   ├── 2_Fares_and_Zones.py
│   ├── 3_Model_Lab.py
│   ├── 4_Fare_Estimator.py
│   └── 5_Insights.py
├── utils/
│   ├── theme.py            # Palette, CSS, hero/KPI/meter components
│   ├── charts.py           # Plotly builders
│   ├── analysis.py         # Formatting and zone ranking
│   ├── data_loader.py      # Cached data and model access
│   └── modeling.py         # Feature construction and predictions
├── data/                   # Aggregates, samples, zone map, model diagnostics
├── models/                 # Saved model, preprocessor, fallback
├── assets/logo.svg
├── build_assets.py
├── requirements.txt
└── .streamlit/config.toml
```

---

# Case study: NYC taxi fare analysis and prediction

## Project overview

This project analyzes nyc taxi trip data to understand taxi demand, fare patterns, passenger behavior, and revenue.

The project covers two main areas:

1. Exploratory data analysis (eda) to identify important patterns in taxi trips and fares.
2. Machine learning to predict the total amount paid for a taxi trip.

The analysis focuses on taxi trips from january 2025.

---

## Dataset

The dataset is the official NYC TLC Yellow Taxi trip data for January 2025
(loaded directly from URL in the notebook, ~3.3M rows, so no data file is
stored in this repo):

`https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet`

The dataset contains information about taxi trips, including:

* Pickup and drop-off times
* Trip distance
* Passenger count
* Pickup and drop-off locations
* Payment type
* Fare amount
* Tips
* Tolls
* Additional charges
* Total amount

### Important features

Some of the main features used in the analysis include:

* Passenger count
* Trip distance
* Ratecodeid
* Pulocationid
* Dolocationid
* Payment type
* Pickup hour
* Pickup day
* Pickup day of week
* Pickup day type

The target variable for the machine learning models is:

`total amount`

---

## Data cleaning and preparation

The dataset was cleaned before performing the analysis and machine learning.

The main preparation steps included:

* Checking for missing values
* Checking for duplicate records
* Checking data types
* Investigating unusual values
* Checking negative fare and total amount values
* Converting pickup and drop-off timestamps to datetime format
* Extracting useful time features

### Feature engineering

The pickup and drop-off datetime columns were used to create:

* Year
* Month
* Day
* Day of week
* Hour
* Weekday/weekend

For example:

`pickup year, pickup month, pickup day, pickup day of week, pickup hour, pickup day type`

These features helped identify how taxi demand and fares change throughout the day and week.

---

# Exploratory data analysis

## Number of taxi trips by hour

The number of taxi trips was analyzed across the 24 hours of the day.

The highest number of trips happened around 6 pm, with more than 253,000 trips. This suggests that taxi demand was particularly high during the evening period.

---

## Number of taxi trips by day of week

The number of trips was compared across the days of the week.

The busiest day recorded more than 590,000 trips. This shows that taxi demand varies depending on the day of the week.

---

## Number of taxi trips by day in january

Daily taxi demand was analyzed throughout january.

January 16 recorded the highest number of trips, with about 134,000 trips. This shows that some days had much higher demand than others.

---

## Weekday vs weekend trips

Taxi trips were divided into weekdays and weekends.

### Results

* Weekday trips: 2.49 million
* Weekend trips: 840,000

There were a lot more taxi trips on weekdays than weekends.

---

## Taxi revenue

The total taxi revenue calculated from the total amount column was $90,335,708.80. This is the total value of all the taxi fares in the dataset.

---

## Revenue by hour

Taxi revenue was analyzed across different hours of the day.

Revenue also changed a lot during the day, with the evening period bringing in a lot of money because so many people took taxis then.

---

## Revenue by day

Daily revenue was analyzed throughout january.

January 16 had the highest daily revenue at about $3.64 million. This matches with the high number of trips on that day.

---

## Average fare by hour

The average fare was calculated for each hour of the day.

The highest average fare happened at 5 am, with $25.67. Even though there are fewer trips at this time, the average fare is higher.

---

## Average fare by day of week

Average fares were also compared for each day of the week.

The highest average fare was on monday, at $21.39. This shows that the busiest day is not always the day with the highest fares.

---

## Passenger count

The number of passengers per trip was analyzed.

Trips with 1 passenger were the most common. This means most taxi trips were individual rides.

---

## Trip distance vs fare

The relationship between trip distance and fare amount was examined.

Usually, longer trips mean higher fares because distance is important for pricing. But the relationship is not perfect because fares can also be affected by time, location, tolls, tips, and other fees.

---

# Machine learning

After finishing the exploratory analysis, machine learning models were built to predict the total amount paid for a taxi trip. The goal was to estimate how much a taxi ride would cost based on information about the trip.

## Features used

The models used features like:

* Vendorid
* Passenger count
* Trip distance
* Ratecodeid
* Store and fwd flag
* Pulocationid
* Dolocationid
* Payment type
* Pickup year
* Pickup month
* Pickup day
* Pickup day of week
* Pickup hour
* Pickup day type

Features that directly make up total amount, like individual fare and tax parts, were not used to avoid making the prediction too easy.

---

# Models used

Three regression models were tested:

## Linear regression

Linear regression was used as a starting point.

Results:

* Mae: $4.13
* Rmse: $7.56
* R²: 0.8589

The model explained about 85.89% of the difference in taxi trip total amounts.

---

## Histgradientboosting

Histgradientboosting was used next to capture more complex patterns.

Results:

* Mae: $2.62
* Rmse: $4.62
* R²: 0.9472

This model was much better than linear regression.

---

## Lightgbm

Lightgbm was also tested because it works well for big datasets and complex relationships.

Results:

* Mae: $2.61
* Rmse: $4.57
* R²: 0.9484

The lightgbm model explained about 94.84% of the total fare differences.

---

# Model comparison

| Model                |   Mae |  Rmse |     R² |
| -------------------- | ----: | ----: | -----: |
| Linear regression    | $4.13 | $7.56 | 0.8589 |
| Histgradientboosting | $2.62 | $4.62 | 0.9472 |
| Lightgbm             | $2.61 | $4.57 | 0.9484 |

The two gradient boosting models did much better than linear regression. The lightgbm model explained about 94.84% of the difference in total taxi fare on the test data.

---

# Overfitting check

The lightgbm model was checked on both the training and testing datasets.

* Training r²: 0.9504
* Testing r²: 0.9484

The difference between the training and testing r² scores is about 0.0020. This means the model works well on new data and is not just memorizing the training examples.

---

# Key findings

Some of the main findings from the project were:

* The dataset generated about $90.34 million in total taxi revenue.
* 6 pm had the highest number of taxi trips, with over 253,000 trips.
* The busiest day had more than 590,000 trips.
* January 16 had the highest daily trip count, about 134,000 trips.
* Weekdays had about 2.49 million trips, weekends had about 840,000 trips.
* January 16 generated about $3.64 million in taxi revenue.
* The highest average fare by hour was at 5 am, $25.67.
* The highest average fare by day was on monday, $21.39.
* Most trips had 1 passenger.
* Lightgbm achieved an r² of 0.9484.
* Lightgbm achieved an mae of $2.61, so its predictions were off by about $2.61 on average.
* The lightgbm training and testing r² scores were very close, with a difference of about 0.002, so there was no serious overfitting.

---

# Conclusion

This project explored nyc taxi trip data from january 2025 to understand taxi demand, pricing, passenger behavior, and revenue patterns. The analysis showed clear differences in taxi demand throughout the day and week. The time with the most trips is not always the time with the highest fares.

Three regression models were tested. Linear regression was a useful starting point, while histgradientboosting and lightgbm gave much better results. The final lightgbm model worked well and did not show signs of overfitting.

Overall, the project shows how python, pandas, plotly, data cleaning, feature engineering, data analysis, regression modeling, and model checking can be used on a big real-world dataset to get useful information.

---

# Technologies used

* Python
* Pandas
* Numpy
* Matplotlib
* Seaborn
* Plotly
* Scikit-learn
* Lightgbm
* Jupyter notebook

---

# Machine learning metrics

* Mae: Mean absolute error (average difference between the real and predicted fare, lower is better)
* Rmse: Root mean squared error (gives more weight to bigger errors, lower is better)
* R² score: Shows how much of the difference in fares is explained by the model (higher is better)

---

# Project workflow

Raw taxi data → data cleaning → feature engineering → exploratory data analysis → feature selection → train/test split → categorical encoding → model training → linear regression → histgradientboosting → lightgbm → model evaluation → overfitting check → final model

---

# Run this project

## Streamlit app

`app.py` is the multi-page dashboard: Overview, Demand & Time, Fares & Zones,
Model Lab, Fare Estimator and Insights. It loads the saved model artifacts from
`models/` (`taxi_model.joblib`, `taxi_preprocessor.joblib`, plus a portable
scikit-learn fallback) and the compact aggregates from `data/`.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notebook

`NYC_Taxi.ipynb` downloads the January 2025 TLC dataset from its URL on first
run (no data file needed in the repo). Requires `pyarrow` for parquet
(included in `requirements.txt`).

```bash
jupyter notebook NYC_Taxi.ipynb
```
