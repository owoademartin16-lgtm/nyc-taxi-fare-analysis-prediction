# Nyc taxi fare analysis and prediction

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

`app.py` is the redesigned fare estimator. It loads `taxi_model.joblib`
(LightGBM) and `taxi_preprocessor.joblib` from this same folder.

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
