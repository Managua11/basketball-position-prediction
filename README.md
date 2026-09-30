Predicting Basketball Positions from Body Measurements

Can we predict a basketball player's position from their body measurements? And does knowing how they play make the prediction better?

This project uses NBA Draft Combine data from 2000–2026 to predict five positions: PG, SG, SF, PF and C.

Results

Using only body measurements, the model predicts the listed position with about 69–71% accuracy.

Adding career NBA statistics improves accuracy to about 75%.

The model works especially well for distinguishing guards from big men. Most mistakes are between neighboring positions, especially SG and SF.

Features	Accuracy	Macro F1
Body measurements	69.4%	68.8%
Career stats	65.9%	64.4%
Body + stats	75.4%	74.4%

The best-performing model was logistic regression.

Data

The body data comes from the NBA Draft Combine and includes:

Height and weight

Wingspan and standing reach

Body fat and hand size

BMI and other derived measurements

For players with enough NBA playing time, career statistics were also added, such as assists, rebounds, blocks and three-point attempts.

The position is the one listed at the Draft Combine. For combined positions such as SG-SF, the first position is used.

How it works

The project has four notebooks:

notebooks/
  01_get_data.ipynb
  02_explore.ipynb
  03_train.ipynb
  04_with_stats.ipynb

predict.py
models/
data/


Run the notebooks in order to download the data, explore it, train the models and compare the results.

Try it
from predict import predict_position, predict_known_player

predict_position(
    height_cm=201,
    weight_kg=98,
    wingspan_cm=210,
    reach_cm=265
)

predict_known_player("Stephen Curry")

Limitations

The position label comes from the Draft Combine, so it may not match the position a player eventually played in the NBA.

The stats model also requires NBA playing time, so it cannot be used for players who have not played in the league yet.

The project is mainly an experiment showing how much body size and playing style can tell us about basketball position.
