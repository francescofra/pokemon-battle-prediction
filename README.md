# Pokémon Battle Prediction

Classical machine-learning approach to predicting the outcome of competitive Pokémon battles from partial battle logs.

This project was developed as our submission to the **2025 Pokémon Battle Prediction Challenge**, organized for the *Foundations of Data Science* course in the MSc programs in Computer Science and Data Science at Sapienza University of Rome.

## The challenge

The task is to predict whether **Player 1 wins a battle**. The dataset contains real human-versus-human matches played on the Pokémon Showdown simulator in the **Generation 1 OverUsed (OU)** competitive format.

For each battle, the available information includes:

- Player 1's complete team of six Pokémon;
- Player 2's known lead Pokémon;
- Pokémon names, levels, types, and base statistics;
- a turn-by-turn summary of the first 30 active turns;
- active Pokémon, remaining HP, status conditions, field effects, stat boosts, and moves;
- the binary target `player_won` for training battles.

Submissions are evaluated using **classification accuracy**.

> **Constraint:** the challenge must be solved exclusively through feature engineering and classical machine-learning models. Neural networks and other deep-learning approaches are not allowed.

## Approach

The raw JSON battle records are transformed into a tabular feature matrix. The engineered features describe several complementary aspects of a battle:

- team composition and aggregate base statistics;
- type distribution and the presence of prominent Generation 1 OU Pokémon;
- HP trajectories and the final observed board state;
- fainted, healthy, unrevealed, and effectively usable Pokémon;
- status conditions, inactivity, switching behavior, and stat boosts;
- physical and special bulk and offensive potential;
- effective Speed and remaining fast Pokémon;
- historical one-on-one matchup statistics extracted from the training battles.

Three base learners are combined in a stacking ensemble:

1. Logistic Regression with standardized features;
2. Random Forest;
3. XGBoost.

A second Logistic Regression model acts as the meta-learner. Random Forest and XGBoost hyperparameters are selected with randomized search.

## Validation

Model performance is estimated using five-fold `GroupKFold` cross-validation, with `battle_id` as the grouping variable. This prevents information associated with the same battle from appearing in both the training and validation portions of a fold.

The stacking ensemble achieved:

| Metric | Result |
|---|---:|
| Mean cross-validation accuracy | **0.8572** |
| Standard deviation | **0.0090** |

Fold accuracies were `0.8640`, `0.8615`, `0.8675`, `0.8450`, and `0.8480`.

## Notebook

The notebook contains the complete workflow:

1. data loading and exploratory analysis;
2. encounter and matchup-statistics extraction;
3. feature engineering;
4. train/test matrix construction;
5. grouped validation setup;
6. Random Forest and XGBoost hyperparameter tuning;
7. stacking evaluation;
8. feature-correlation and importance visualizations;
9. final training and submission generation.

The final output is a `submission.csv` file containing:

```text
battle_id,player_won
```

## Running the project

The notebook was designed to run in a Kaggle environment with the competition data attached.

Required Python packages:

```bash
pip install numpy pandas matplotlib scipy scikit-learn xgboost
```

The dataset directory must contain:

```text
train.jsonl
test.jsonl
sample_submission.csv
```

Run all notebook cells from top to bottom. The hyperparameter searches and nested stacking evaluation are computationally expensive and may take several minutes.


## Data availability

The competition data are **not included** in this repository. They remain subject to the terms and access rules of the original challenge. To reproduce the analysis, obtain the data through the course or competition platform and attach them to the notebook environment.

