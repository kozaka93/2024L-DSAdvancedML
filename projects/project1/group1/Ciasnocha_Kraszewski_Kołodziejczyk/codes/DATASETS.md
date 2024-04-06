# Datasets for binary classification
Summary of datasets chosen for the project. In total 9 datasets are presented, including 3 small ones (<= 10 features) and 3 large ones (> 10 features).
Each entry includes link where one can find the dataset, number of features and instances after the preprocessing, and the details how a particular dataset was preprocessed.
Code used for preprocessing can be found in `preprocessing.ipynb` notebook.

## Small Datasets

### 1. [National Health and Nutrition Health Survey 2013-2014 (NHANES) Age Prediction Subset](https://archive.ics.uci.edu/dataset/887/national+health+and+nutrition+health+survey+2013-2014+(nhanes)+age+prediction+subset)
- **Features:** 6
- **Instances:** 2278
- **Objective:** Predict age group (senior/non-senior) based on medical features.

#### Preprocessing applied:
- drop SEQN serial number
- drop collinear variables: 
  - LBXIN (0.55 correlation with BMXBMI)
  - LBXGLU (0.69 correlation with LBXGLT)
- transform age group to binary target (senior/non-senior):
  - 0: Adult
  - 1: Senior

### 2. [Ajwa or Medjool](https://archive.ics.uci.edu/dataset/879/ajwa+or+medjool)
- **Features:** 3
- **Instances:** 20
- **Objective:** Predict date fruit species (Ajwa/Medjool) based on physical dimensions, weight, and calories.

#### Preprocessing applied:
- drop columns:
  - color (it's 1-to-1 with target)
  - calories (highly correlated with date weight and date length)
  - date length (highly correlated with dat weight and pit length)
- transform species to binary target (Ajwa/Medjool):
  - 0: Ajwa
  - 1: Medjool

### 3. [Fertility Dataset](https://archive.ics.uci.edu/dataset/244/fertility)
- **Features:** 12
- **Instances:** 100
- **Objective:** Predict fertility based on behavioral factors (discrete numerical values based on subject's response).

#### Preprocessing applied:
- one-hot encode the season feature
- transform fertility to binary target (normal/altered):
  - 0: Normal
  - 1: Altered

Note that the dataset is small and imbalanced, with only 12 instances of altered fertility.

## Large Datasets

### 1. [Mice Protein Expression](https://archive.ics.uci.edu/dataset/342/mice+protein+expression)
- **Features:** 72
- **Instances:** 1077 (has missing values)
- **Objective:** Predict whether mice have Down syndrome based on some proteins in the brain.

#### Preprocessing applied:
- originally 49 columns had missing values:
  - dropped 3 instances with lots of missing columns, afterwards only 9 columns had missing values
  - dropped 5 columns with more than 10% missing values
  - for the last 4 columns, filled missing values with the overall mean of the column as there was no significant difference between means of the two classes
  - dropped categorical features other than Genotype
  - dropped highly correlated columns (r > 0.8)
  - transformed Genotype binary target (Control vs. Ts65Dn):
    - 0: Control
    - 1: Ts65Dn

### 2. [Mushroom Dataset](https://archive.ics.uci.edu/dataset/73/mushroom)
- **Features:** 117
- **Instances:** 8124
- **Objective:** Real data on mushrooms, predict whether a mushroom is edible or poisonous.

#### Preprocessing applied:
- one-hot encode all features
- transform class to binary target (edible/poisonous):
  - 0: Edible
  - 1: Poisonous

### 3. [Room Occupancy Estimation](https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation)
- **Features:** 11
- **Instances:** 10129
- **Objective:** Estimate room occupancy (0, 1, 2, 3 people) based on sensor data.

#### Preprocessing applied:
- drop date and time columns
- drop highly correlated columns (r > 0.8)
- transform occupancy to binary target (none or one person/two or three people):
  - 0: Zero or one person
  - 1: Two or three people

### 4. [Taiwanese Bankruptcy Prediction](https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction)
- **Features:** 70
- **Instances:** 6819
- **Objective:** Predict whether a company will go bankrupt based on business-related data.

#### Preprocessing applied:
- drop highly correlated columns (r > 0.8)
- transform bankruptcy to binary target (non-bankrupt/bankrupt):
  - 0: Non-bankrupt
  - 1: Bankrupt

Note that the dataset is imbalanced, with only 220/6819 instances of bankrupt companies.

### 5. [Estimation of Obesity Levels](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)
- **Features:** 27
- **Instances:** 2111 (preprocessing required)
- **Objective:** Predict obesity level based on patient's survey. Data conversion required for binary meaning (e.g., normal+overweight vs. obese).

#### Preprocessing applied:
- one-hot encode following features: Gender, MTRANS, CAEC, CALC
- replace 'yes'/'no' with 1/0
- transform obesity level to binary target (underweight+normal+overweight/obese):
  - 0: Insufficient_Weight, Normal_Weight, Overweight_Level_I, Overweight_Level_II
  - 1: Obesity_Type_I, Obesity_Type_II, Obesity_Type_III

### 6. [Wine Color](https://archive.ics.uci.edu/dataset/186/wine+quality)
- **Features:** 12
- **Instances:** 6497
- **Objective:** Predict wine color (red/white) based on chemical properties.

#### Preprocessing applied:
- introduce binary target (red/white):
  - 0: Red
  - 1: White