from flask import Flask, render_template
import pandas as pd
from lifetimes import BetaGeoFitter
from lifetimes.utils import summary_data_from_transaction_data

# Load data from CSV files
orders = pd.read_csv('data/olist_orders_dataset.csv')
items = pd.read_csv('data/olist_order_items_dataset.csv')
cust = pd.read_csv('data/olist_customers_dataset.csv')

# Merge data and drop duplicates
orders = pd.merge(orders, cust[['customer_id', 'customer_unique_id']], on='customer_id')
items.drop_duplicates('order_id', keep='first', inplace=True)
transaction_data = pd.merge(orders, items, 'inner', 'order_id')
transaction_data = transaction_data[['customer_unique_id', 'order_purchase_timestamp', 'price']]

# Convert timestamp to date
transaction_data['date'] = pd.to_datetime(transaction_data['order_purchase_timestamp']).dt.date
transaction_data = transaction_data.drop('order_purchase_timestamp', axis=1)

# Calculate summary data for Lifetimes model
summary = summary_data_from_transaction_data(transaction_data, 'customer_unique_id', 'date', monetary_value_col='price')

# Select data from customers who made at least one repeat purchase
df = summary[summary['frequency'] > 0]

# Function for data analysis and churn prediction
def analyze_churn():
    # Check if the dataframe is not empty
    if not df.empty:
        # Fit the BG/NBD model and calculate probability of being alive
        bgf = BetaGeoFitter(penalizer_coef=0.00)
        bgf.fit(df["frequency"], df["recency"], df["T"])
        df["prob_alive"] = bgf.conditional_probability_alive(df["frequency"], df["recency"], df["T"])

        # Predict churn and identify high-risk customers
        df["churn"] = ["churned" if p < 0.1 else "not churned" for p in df["prob_alive"]]
        df["churn"] = df["churn"].where((df["prob_alive"] >= 0.1) & (df["prob_alive"] < 0.2), "high risk")

        # Return a filtered DataFrame with high-risk customers
        return df[df["churn"] == "high risk"]
    else:
        # Return an empty DataFrame if there's no data for analysis
        return pd.DataFrame(columns=["customer_unique_id", "frequency", "recency", "T", "prob_alive", "churn"])

# Flask app instance
app = Flask(__name__)

# Route for the main page
@app.route("/")
def main():
    high_risk_df = analyze_churn()
    return render_template("index.html", high_risk_df=high_risk_df)

if __name__ == "__main__":
    app.run(debug=True)
