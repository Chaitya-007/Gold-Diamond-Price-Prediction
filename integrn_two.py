# Importing necessary libraries
import tkinter as tk
from tkinter import messagebox
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

# Load the dataset (assuming you have a CSV file named 'modified_gold.csv' with columns 'Year' and 'Price')
data = pd.read_csv('modified_gold.csv')

# Convert 'Year' column to datetime type
# data['Year'] = pd.to_datetime(data['Year'])

# Extract year from the 'Date' column and create a new column 'Year'
# data['Year'] = data['Date'].dt.year

# Splitting data into features (X) and target variable (y)
X = data[['Year']]
X.columns = ['Year']
y = data['Price']

# Visualizing the gold prices over the years
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Gold Prices')
plt.title('Gold Prices Over the Years')
plt.xlabel('Year')
plt.ylabel('Price')
plt.grid(True)
plt.legend()
plt.show()

# Splitting the gold dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the linear regression model for gold
model = LinearRegression()
model.fit(X_train, y_train)

# Function to predict gold price for a given year
def predict_gold_price(year):
    year = [[year]]
    predicted_price = model.predict(year)
    return predicted_price[0]

# Function to handle button click event for gold prediction
def predict_gold_button_click():
    year = int(entry_year.get())
    predicted_price = predict_gold_price(year)
    result_label.config(text=f"Predicted gold price for {year}: ₹{predicted_price:.2f}")


# Load the dataset for diamonds (assuming you have a CSV file named 'diamond_prices.csv' with columns 'Year' and 'Price')
diamond_data = pd.read_csv('diamond_prices.csv')

# Convert 'Year' column to datetime type (uncomment if 'Year' is in datetime format)
# diamond_data['Year'] = pd.to_datetime(diamond_data['Year'])

# Splitting data into features (X_diamond) and target variable (y_diamond)
X_diamond = diamond_data[['Year']]
X_diamond.columns = ['Year']
y_diamond = diamond_data['Price']

# Visualizing the diamond prices over the years
plt.figure(figsize=(10, 6))
plt.scatter(X_diamond, y_diamond, color='red', label='Diamond Prices')
plt.title('Diamond Prices Over the Years')
plt.xlabel('Year')
plt.ylabel('Price')
plt.grid(True)
plt.legend()
plt.show()

# Splitting the diamond dataset into training and testing sets
X_train_diamond, X_test_diamond, y_train_diamond, y_test_diamond = train_test_split(X_diamond, y_diamond, test_size=0.2, random_state=42)

# Training the linear regression model for diamonds
model_diamond = LinearRegression()
model_diamond.fit(X_train_diamond, y_train_diamond)

# Function to predict diamond price for a given year
def predict_diamond_price(year):
    year = [[year]]
    predicted_price = model_diamond.predict(year)
    return predicted_price[0]

# Function to handle button click event for diamond prediction
def predict_diamond_button_click():
    year = int(entry_year_diamond.get())
    predicted_price_diamond = predict_diamond_price(year)
    result_label_diamond.config(text=f"Predicted diamond price for {year}: ₹{predicted_price_diamond:.2f}")


# Creating main window
root = tk.Tk()
root.title("Price Prediction")
root.geometry("400x400")

# Load and display prediction image
prediction_image = Image.open("prediction.jpeg")
# prediction_image = prediction_image.resize((200, 200), Image.ANTIALIAS)
prediction_photo = ImageTk.PhotoImage(prediction_image)
prediction_label = tk.Label(root, image=prediction_photo)
prediction_label.pack()

# Question label
question_label = tk.Label(root, text="Which price do you want to predict?", font=("Arial", 14))
question_label.pack(pady=(20, 10))

# Radio button selection for prediction
var = tk.StringVar()
var.set("Gold")

gold_radio = tk.Radiobutton(root, text="Gold Price Prediction", variable=var, value="Gold", font=("Arial", 12), command=select_prediction)
gold_radio.pack()

diamond_radio = tk.Radiobutton(root, text="Diamond Price Prediction", variable=var, value="Diamond", font=("Arial", 12), command=select_prediction)
diamond_radio.pack()

# Frame for gold prediction
gold_frame = tk.Frame(root)

# Gold prediction button
gold_button = tk.Button(gold_frame, text="Predict Gold Price", command=predict_gold_price, bg="#4caf50", fg="white", font=("Arial", 12))
gold_button.pack(pady=10)

# Frame for diamond prediction
diamond_frame = tk.Frame(root)

# Diamond prediction button
diamond_button = tk.Button(diamond_frame, text="Predict Diamond Price", command=predict_diamond_price, bg="#4caf50", fg="white", font=("Arial", 12))
diamond_button.pack(pady=10)

root.mainloop()
