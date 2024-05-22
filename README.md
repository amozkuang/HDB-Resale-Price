# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 2: Singapore Housing Data and Kaggle Challenge

### Group: 1

### Name: Amoz Kuang / Gilbert Hartato / Syahiran Rafi

---
[Note: This project has been revised on May 22, 2024 to include a competitive model (Random Forest), and deployment of the predicion app.]

### PROBLEM STATEMENT

When it comes to property prices in Singapore, the on-ground sentiments we often hear are — "so expensive", "so far", "is it worth the price?"

Yet, there still seems to be a preference amongst some to stay in the central region due to convenience and general accessibility. In our study, we examine how the price of resale HDB flats, particularly in the central region, are influenced by various factors like floor area, age of HDB flat, maximum floor level, proximity to amenities and public transport connectivity.

Ultimately, we aim to address the following question: **"Are resale prices of central region HDBs influenced primarily by its location?"**

In doing so, we hope to empower our target audience to make more calculated and informed decisions on housing whether it's young couples buying their first flats, or older families looking to sell their flats

---

### TARGET CLIENT PROFILE

**Mr. and Mrs. Tan - the typical Singapore nuclear family**

- 2 children (age 2 and 5)
- Currently living in a 4-room HDB flat in Tampines
- Intend to upgrade to a 5-room HDB flat
- Central location is preferred, but not a must
- Accessibility by public transport is highly important
- Less than $550k budget

---

### DATASETS

The following datasets correpond to HDB resale information in Singapore from 2012 to 2021:

* __train.csv__
  - Full dataset with 150,634 rows and 77 columns
* __test.csv__
  - Smaller dataset with 16,737 rows and 77 columns
* __train_clean.csv__ and __test_clean.csv__
  - Changed the column name to lowercase
  - Changed all NaN values to 0
  - Changed 'residential', 'commercial','market_hawker','multistorey_carpark','precinct_pavilion' (Y=1, N=0)
  - Dropped columns (all latitudes and longitutde-related, all rental-related, 'precinct_pavilion', 'residential', 'commercial', 'block', 'street_name', 'address', 'postal', 'bus_stop_name', 'Tranc_YearMonth', 'Tranc_Month', 'floor_area_sqm', 'full_flat_type', 'lease_commence_date', 'year_completed', 'price_per_sqft', 'pri_sch_affiliation')
* __train_edavis.csv__ and __test_edavis.csv__
  - Changed the column name to lowercase
  - Changed all NaN values to 0
  - Changed 'residential', 'commercial','market_hawker','multistorey_carpark','precinct_pavilion' (Y=1, N=0)
  - Dropped columns (all latitudes and longitutde-related, all rental-related, 'precinct_pavilion', 'residential', 'commercial', 'block', 'street_name', 'address', 'postal', 'bus_stop_name', 'Tranc_YearMonth', 'Tranc_Month', 'floor_area_sqm', 'full_flat_type', 'lease_commence_date', 'year_completed', 'price_per_sqft', 'pri_sch_affiliation')
  - Defining functions - 'central_regions' and 'non_central_regions' and 'other'

---

### DATA DICTIONARY


| Feature                  | Type      | Description                                                                     |
| -------------------------- | ----------- | --------------------------------------------------------------------------------- |
| town                     | *integer* | Represents mature estate = 1, non-mature estate = 0                             |
| floor_area_sqft          | *float*   | Floor area of the resale flat unit in square feet                               |
| hdb_age                  | *integer* | Number of years from lease_commence_date to present year                        |
| max_floor_lvl            | *integer* | Highest floor of the resale flat                                                |
| mid_storey               | *integer* | Median value of storey_range                                                    |
| 1room_sold               | *integer* | Number of 1-room residential units in the resale flat                           |
| 2room_sold               | *integer* | Number of 2-room residential units in the resale flat                           |
| 3room_sold               | *integer* | Number of 3-room residential units in the resale flat                           |
| 4room_sold               | *integer* | Number of 4-room residential units in the resale flat                           |
| 5room_sold               | *integer* | Number of 5-room residential units in the resale flat                           |
| exec_sold                | *integer* | Number of executive type residential units in the resale flat block             |
| multigen_sold            | *integer* | Number of multi-generational type residential units in the resale flat block    |
| studio_apartment_sold    | *integer* | Number of studio apartment type residential units in the resale flat block      |
| cutoff_point             | *integer* | PSLE cutoff point of the nearest secondary school                               |
| affiliation              | *integer* | Boolean value if the nearest secondary school has an primary school affiliation |
| sec_sch_name             | *integer* | Represents: Top secondary school = 1, Other secondary school = 0                |
| pri_sch_name             | *integer* | Represent: Top primary school = 1, Other primary school = 0                     |
| mrt_nearest_distance     | *float*   | Distance (in metres) to the nearest MRT station                                 |
| pri_sch_nearest_distance | *float*   | Distance (in metres) to the nearest primary school                              |
| sec_sch_nearest_dist     | *float*   | Distance (in metres) to the nearest secondary school                            |
| hawker_within_2km        | *float*   | Number of hawker centres within 2 kilometres                                    |
| region                   | *integer* | Represents Central Region = 1, Outside Central Region = 0                       |
| planning_area            | *string*  | Government planning area that the flat is located                               |
| flat_type                | *string*  | Type of the resale flat unit, e.g. 3 ROOM                                       |
| flat_model               | *string*  | HDB model of the resale flat, e.g. Multi Generation                             |

---

### KEY INSIGHTS

_- Yes, resale prices of central region HDBs seem to be influenced primarily by its location._

- Secondarily, buyers may be drawn to the generally larger floor areas of central HDBs. However, this may also be found in non-central mature estates._

_- Other factors such as proximity to top schools, malls, hawker centres and connectivity via public transport don’t seem to affect resale prices in the central region as much._

---

### RECOMMENDATIONS TO CLIENT

- Client may consider towns that border the central region
  - More budget-friendly (resale price generally below $500k)
- Towns such as Serangoon and Ang Mo Kio also offer good public transport connectivity.
  - Serangoon has a bus and an MRT interchange
  - AMK has a bus interchange, and is one MRT stop away from an MRT interchange (Bishan)

---

### OUR PREDICTIVE MODEL


| Model             | cross_val_score(train) | cross_val_score(test) |
| ------------------- | ------------------------ | ----------------------- |
| Linear Regression | 0.873                  | -7.352e+23            |
| Ridge Regression  | 0.873                  | 0.872                 |
| Lasso Regression  | 0.872                  | 0.871                 |

- **Ridge regression** is selected for this modelling:

  - Highest cross_val_score among the 3 models
  - Comparing test and train score, ridge regression provide the best fit among the 3 models
- Used to predict the prices of resale HDB flats

  - Coefficient of determination, R^2 score: 0.87
  - Mean absolute percentage error, MAPE score: 9.4%

### FUTURE RECOMMENDATION FOR MODELLING


| Model Limitation                                                                                                   | Possible Solutions                                                 |
| -------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| Latest transactions in the dataset took place in 2021                                                              | Include transactions till 2023 to reflect the latest resale prices |
| Improved infrastructure such as new MRT lines and URA urban planning within certain towns may affect resale prices | Include “developing_towns” as an additional feature in dataset   |
| Transaction volume alone may not be a sufficient indicator of supply & demand                                      | Include “time_taken_to_sell” as an additional feature in dataset |

---

---

---
