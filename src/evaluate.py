
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class EvaluateModel():

    def __init__(self):
        return

    def evaluate_model(self, y_true, y_pred):
        print("\nModel Performance:")
        print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_true, y_pred):.2f}")
        print(f"Mean Squared Error (MSE): {mean_squared_error(y_true, y_pred):.2f}")
        print(f"R² Score: {r2_score(y_true, y_pred):.4f}")

    # Function to plot predictions
    def plot_all_predictions(
            self,
            dates, y_actual,
            train_dates, test_dates,
            y_train_pred, y_test_pred,
            title
    ):
        plt.figure(figsize=(12, 6))

        # Plot actual Bitcoin price in black
        plt.plot(dates, y_actual, label='Actual Bitcoin Price', color='black', linewidth=2)

        # Plot Stacking Regressor predictions in blue
        plt.plot(train_dates, y_train_pred,  label='Voting Regressor on Training Data', color='orange',  linestyle='dashed', linewidth=2)

        plt.plot(test_dates, y_test_pred,  label='Voting Regressor on Test Data', color='red', linestyle='dashdot', linewidth=2)

        # Calculate the +/- 5% band around the actual prices
        upper_bound = [price * 1.10 for price in y_actual]
        lower_bound = [price * 0.90 for price in y_actual]
        plt.fill_between(dates, lower_bound, upper_bound, color='green', alpha=0.2, label="±10% Bitcoin Price Band")


        # Labels, title, and legend
        plt.xlabel("Date")
        plt.ylabel("Bitcoin Price (USD)")
        plt.title(title)
        plt.legend()
        plt.grid(True)

        # Show the plot
        plt.savefig('../plots/prediction_vs_actual.png', dpi=300)