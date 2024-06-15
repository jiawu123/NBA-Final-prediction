# NBA-Final-prediction
# Predicting NBA Finals Outcomes Using Machine Learning

*This project aims to predict the outcome of NBA Finals games using machine learning models. We analyze team statistics from multiple datasets, focusing on advanced statistics and per 100 possession statistics, to determine which metrics significantly influence game outcomes. We then use this information to predict the winner of the 2024 NBA Finals.*
## Challenge Goals

### Goal 1: Utilize Multiple Datasets
**Achievement**: Three datasets were used together: advanced statistics, per 100 possession statistics, and historical NBA Finals data.
**How**: We merged the advanced stats and per 100 possession stats for each year and combined them with the finals data. This allowed us to have a comprehensive dataset containing both regular season and finals performance metrics.

### Goal 2: Implement Advanced Machine Learning Techniques
**Achievement**: Compared at least three different machine learning algorithms and used feature selection.
**How**:
1. We trained and compared Logistic Regression, Random Forest, and Gradient Boosting models.
2. Used Recursive Feature Elimination with Cross-Validation (RFECV) to select the most predictive features for each model.
3. Evaluated each model using cross-validation scores and selected the best-performing model (Random Forest) for the final prediction.
   ## Data Setting and Methods

### Data Setting:
- **Source**: All datasets were sourced from [Basketball Reference](https://www.basketball-reference.com/). The specific tables used were:
  - **Advanced Statistics**: Yearly advanced stats for all NBA teams from 2010 to 2024.
  - **Per 100 Possession Statistics**: Yearly per 100 possession stats for all NBA teams from 2010 to 2024.
  - **NBA Finals Data**: Historical data of teams that reached the NBA Finals and the winning team from 2010 to 2023.
- **Data Volume**: The combined dataset includes:
  - **18 tables**: 13 years of advanced stats, 13 years of per 100 possession stats, and NBA Finals data.
  - **Columns**: Approximately 50 columns after merging.
  - **Rows**: 260 rows in total.
- **Why**: These datasets provide a comprehensive view of team performance, enabling the analysis of factors influencing NBA Finals outcomes.
- **How**: 
  1. **Data Cleaning**: Removed empty and unnamed columns to ensure data quality.
  2. **Data Merging**: Merged advanced stats and per 100 possession stats on 'Year' and 'Team' columns. Further merged the combined stats with finals data on 'Year' and 'Team' columns.

### Methods:
1. **Data Cleaning**:
   - Removed empty and unnamed columns.
   - Ensured consistency in column names and formats.
2. **Data Merging**:
   - Merged advanced stats and per 100 possession stats on the 'Year' and 'Team' columns.
   - Merged the combined stats with the finals data on the 'Year' and 'Team' columns.
3. **Feature Selection**:
   - Used Recursive Feature Elimination with Cross-Validation (RFECV) to identify the most predictive features for each model.
4. **Model Training**:
   - Trained and evaluated Random Forest models using cross-validation.
5. **Prediction**:
   - Predicted the 2024 NBA Finals winner using the trained model. model.*

### Research Question 1: Which team statistics are most predictive of winning the NBA Finals?
The Random Forest model identified PW, NRtg, and ORB% as the most predictive features.
#### Feature Importance Bar Chart:
To visualize the importance of each selected featur Iwe plotted a bar chart of feature importances. This helps in understanding which features the model considers most significant for predicting NBA Finals outcometa.*
### Research Question 2: How does the correlation between different team statistics influence game outcomes?
The correlation heatmap shows significant positive correlations between ORtg and NRtg with winning, and a negative correlation with TOV%.

Correlation Heatmap:
The correlation heatmap visualizes the relationships between all features, highlighting strong positive and negative correlations. This helps in understanding how different metrics interact and influence game outcomes.
### Research Question 3: Can machine learning models accurately predict the winner of the NBA Finals based on historical data?

The Random Forest model achieved an average cross-validated accuracy score of 87%. The model predicted the Boston Celtics as the winner of the 2024 NBA Finals with a probability of 15%.

#### Winning Probabilities Bar Chart:
We visualized the predicted probabilities of winning for the 2024 NBA Finals teams. This bar chart provides a clear comparison of the model's confidence in each team's chances of winning. The predicted probabilities for the Boston Celtics and the Dallas Mavericks were 0.15 and 0.08, respectively. 

The close probabilities indicate that the prediction is highly competitive and suggests that the model finds both teams to be strong contenders. The relatively low probability for the predicted winner, Boston Celtics, implies that while they are favored, their advantage is marginal. This could be due to several factors such as the variability in game outcomes, the strength of the opposing team, and possible unforeseen events during the finals.

The Boston Celtics' slight outperformance in the model's prediction could be attributed to better historical data metrics in key areas identified by the model, such as PW (Pythagorean Wins), NRtg (Net Rating), and ORB% (Offensive Rebound Percentage). These metrics suggest that the Celtics have had a more consistent and effective pe  seasons compared to the Mavericks.

However, the closeness of the probabilities highlights the inherent uncertainty and competitiveness in predicting sports outcomes, indicating that while the Celtics may have a slight edge, the actual game results could vary significantly based on real-time performance and other dynamic factors.
## Implications and Limitations

### Implications:
1. **For Coaches and Analysts**: The analysis can help in understanding key performance metrics that influence game outcomes. By focusing on metrics like PW, NRtg, and ORB%, coaches can devise strategies to improve these areas and enhance their team's chances of winning.
2. **For Teams**: Insights from the model can inform strategies to improve chances of winning. Teams can use the model to identify strengths and weaknesses and tailor their training and game plans accordingly.
3. **For Fans and Bettors**: Provides a data-driven approach to predicting game outcomes. Fans can use the insights to better understand game dynamics, and bettors can make more informed decisions based on the model's predictions.

### Limitations:
1. **Pool Strength Disparities**: The strength of the pools (Eastern vs. Western Conference) can be largely different. If a team like the Boston Celtics is in a weaker pool, their data might show better performance metrics compared to teams in a stronger pool. This can skew the model's predictions as it doesn't account for the relative strength of opponents.
2. **Lack of Individual Player Metrics**: The dataset lacks detailed metrics on the performance of individual players, especially during clutch times. Players like Tatum and Jalen on the Celtics or Kyrie and Dončić on the Mavericks have significant impacts on game outcomes, but these are not directly captured in the data. The absence of these metrics limits the model's accuracy in predicting close game outcomes.
3. **Data Quality and Completeness**: The data used might not capture all aspects of team performance and external factors such as injuries, player trades, and coaching changes. These factors can significantly influence game outcomes but are not reflected in the datasets.
4. **Model Generalization**: The model's predictions are based on historical data and may not generalize well to future games with different conditions. Changes in team dynamics, strategies, and player performances over time can lead to deviations from the model's predictions.
5. **Feature Selection Bias**: The reliance on selected features might overlook other important but less obvious metrics. The model might miss out on capturing the full complexity of the game due to the limited number of features considered.

These limitations suggest that while the model provides useful insights, it should not be the sole basis for critical decisions. Coaches, analysts, and bettors should consider these factors and use the model's predictions as one of several tools in their decision-making process.
