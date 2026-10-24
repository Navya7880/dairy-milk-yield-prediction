# Viva Preparation: Dairy Milk Yield Prediction

Use short, accurate answers. The project currently uses synthetic demonstration data, not real dairy-farm evidence.

## 1. What does this project do?
It predicts daily milk yield per cow from seven supported inputs, manages cattle observations, analyzes saved records, and explains a held-out evaluation. It is a decision-support demonstration.

## 2. What is the project title?
A Multiple Linear Regression Model for Dairy Milk Yield Prediction.

## 3. Why is the project web based?
Streamlit provides a beginner-friendly browser interface directly in Python. It lets the UI invoke data processing, database, and model modules without a separate HTTP API.

## 4. What is the dependent variable?
`milk_yield_liters`, meaning liters of milk per cow per day.

## 5. What are the independent variables?
Breed, age in months, lactation number, body weight in kilograms, daily feed in kilograms, parity stage, and days in milk.

## 6. What are the units of the prediction?
Liters per cow per day.

## 7. Why was Multiple Linear Regression selected?
It matches a continuous numerical target, is interpretable as a linear baseline, uses a standard scikit-learn workflow, and is suitable for the project's academic scope.

## 8. Does this prove that regression coefficients cause yield changes?
No. The coefficients describe associations in the demonstration data; they do not prove causation.

## 9. Where did the dataset come from?
The included CSV is generated locally by a seeded synthetic generator in `src/data_processing.py`. It is not claimed to be a public real-farm dataset.

## 10. Why is synthetic data used?
The empty workspace had no suitable real dairy dataset. Synthetic data allowed a complete, honest application demonstration while clearly documenting that real-world performance is unknown.

## 11. What does the target represent?
A single simulated daily yield value in liters per cow.

## 12. What is `days_in_milk`?
The day within the lactation period. It is limited to 1 to 305 in the dataset and form.

## 13. How are missing values handled?
Numeric features use median imputation and categorical features use most-frequent imputation inside the scikit-learn pipeline.

## 14. Why is preprocessing inside a pipeline?
The same fitted preprocessing is reused for test data and later predictions. This prevents manually processing new inputs differently and prevents retraining leakage.

## 15. How are categories encoded?
One-hot encoding with `handle_unknown="ignore"` converts breed and parity stage to binary feature columns.

## 16. What is data leakage?
Using test or future information while fitting preprocessing or the model. Here preprocessing and the model fit only on training rows.

## 17. How is the data split?
An 80/20 train/test split with seed 42.

## 18. What is overfitting?
When a model learns training patterns too specifically and performs worse on unseen data. Held-out metrics are intended to reveal this, but they cannot validate real-farm transfer.

## 19. What is R²?
The proportion of held-out target variance explained by the model relative to a horizontal mean baseline. It is not a percentage of correct predictions.

## 20. What is MAE?
Mean Absolute Error: the average absolute difference between predicted and actual yield in liters.

## 21. What is RMSE?
Root Mean Squared Error: similar to MAE but more strongly penalizes large errors, also in liters.

## 22. What are the current demo metrics?
Held-out R² is 0.884, MAE is 0.9062 L/cow/day, and RMSE is 1.1887 L/cow/day. The test set has 160 rows.

## 23. How many rows train and test?
640 rows train and 160 rows test.

## 24. What is the mean baseline?
The baseline always predicts training-set mean yield. It gives context for whether the regression is better than a very simple estimate.

## 25. What is the mean baseline MAE/RMSE?
The demo baseline is 2.8926 MAE and 3.5109 RMSE in liters per cow per day. It is worse than the regression on held-out data.

## 26. What does Pandas do?
Pandas loads and validates tables and handles filtering, CSV operations, and record frames.

## 27. What does NumPy do?
NumPy supplies vectorized numeric generation and calculations used by the synthetic dataset and evaluation code.

## 28. What does scikit-learn do?
It provides preprocessing, pipelines, train/test splitting, linear regression, and metrics.

## 29. What does Plotly do?
Plotly creates interactive distributions, trend charts, category summaries, and predicted-versus-actual charts.

## 30. What does Streamlit do?
It renders the interface, routes sidebar navigation, manages form interactions and reruns, and directly invokes Python backend functions.

## 31. What does Joblib do?
Joblib serializes the complete fitted preprocessing-and-model pipeline so it can be loaded for later predictions.

## 32. How does prediction work?
The app validates inputs against the shared feature names and ranges, creates one DataFrame row in the exact order, loads the pipeline, and calls `predict`.

## 33. Can missing model files be recovered safely?
The app reports that training must be run. It does not silently generate or invent a model during prediction.

## 34. How are cattle IDs kept unique?
The database primary key enforces uniqueness. The UI and import path also validate IDs before writes.

## 35. How does SQLite fit the architecture?
SQLite is the default local database. It persists records in `data/dairy_records.db` during local use.

## 36. How does PostgreSQL support work?
When `DATABASE_URL` is set, storage opens PostgreSQL and uses the same parameterized cattle-record operations. It uses `psycopg[binary]`.

## 37. Does Streamlit Cloud guarantee SQLite persistence?
No. Managed redeployments or restarts may not preserve local files. Persistent PostgreSQL is required for a durable deployment.

## 38. What is a safe database query?
A parameterized query keeps values separate from SQL text. The storage module uses placeholders and does not build SQL by concatenating user values.

## 39. What CSV validation is implemented?
The import requires exact columns, valid numeric values, required values, unique/non-empty cattle IDs, and a maximum file size of 2 MB.

## 40. How does CSV export work?
Pandas serializes a visible filtered DataFrame to CSV bytes and Streamlit creates a download button.

## 41. Can users edit records?
Yes. The app reads an existing record, pre-fills the form, updates it through the storage module, and refreshes cached records.

## 42. Why is deletion confirmed?
Deletion is permanent. The app requires an explicit checkbox before invoking the delete operation.

## 43. How is analytics based on real data?
Charts and averages use only saved database rows. Empty and invalid-filter states are handled without fabricated values.

## 44. Is total recorded production always shown?
No. The dashboard sums only valid observed yields and labels it as record-observation yield. It does not claim all cows or production periods are covered.

## 45. How are Streamlit reruns handled?
Reruns read database-backed state again and only cache the short-lived record read. Mutating operations are not cached.

## 46. Does the app contain authentication?
No. It is a shared demo. Production use would require identity, roles, authorization, audit logs, and deployment hardening.

## 47. What are the model limitations?
It is linear, trained on one synthetic dataset, has a limited feature set, and cannot account for weather, disease, feed composition, or management effects absent from the data.

## 48. How can a prediction be explained to a stakeholder?
Describe it as an estimate based on the entered values and the demonstrated relationship. Mention that a real farm must compare it with validated cattle observations before decisions.

## 49. What is genuinely implemented?
The working UI, CRUD, import/export, prediction, persisted pipeline, evaluation metadata, charts, filters, tests, and deployment instructions are implemented.

## 50. What is not implemented?
Real validated dairy data, field experiments, authentication, multi-tenant authorization, legal/compliance controls, and a production cloud deployment are not claimed.

## 51. What future enhancements would you propose?
Use licensed real-farm longitudinal data, validate generalization across farms, include useful missing features, compare models, add verified actual outcomes and audit history, and introduce role-based access.

## 52. How is the project deployed?
Streamlit Community Cloud can run `app.py` with `requirements.txt`. The database uses `DATABASE_URL` from app Secrets. No live deployment URL or deployment success is claimed unless it has been verified.

## 53. How does a typical user flow work?
Open the dashboard, add or import records, filter analytics, enter prediction inputs, inspect the estimated yield, and review model metrics and limitations.

## 54. What is the difference between a prediction and an observed yield?
An observed yield comes from saved cattle measurements. A prediction is an estimate made by the learned pipeline and is clearly labeled as such.

## 55. Why does the app reject negative or extreme inputs?
Schema limits and non-empty categories reduce invalid or unsafe demonstration inputs. They are validation rules, not biological proof.

## 56. What would invalidate the current metrics?
Using the model on breeds, husbandry, feeds, climates, or lactation conditions outside the synthetic generator, or treating synthetic performance as real evidence.

## 57. How was the model saved?
`joblib.dump` serialized the complete pipeline, not only coefficients. The app reloads the same object and validates that it provides prediction behavior.

## 58. What runs during model training?
Training creates preprocessing, fits both preprocessing and regression on train rows only, evaluates test rows, calculates baseline metrics, saves the pipeline, and writes JSON metadata.

## 59. How do app and training features stay consistent?
Both use `src.data_processing.FEATURES`. Prediction validation and form generation use the same feature contract.

## 60. What is the main academic conclusion?
A Multiple Linear Regression pipeline can be demonstrated transparently and reproducibly, but conclusions and deployment readiness remain limited until real, licensed, representative dairy data are independently validated.
