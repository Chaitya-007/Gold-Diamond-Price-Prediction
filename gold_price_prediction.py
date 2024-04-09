# Importing necessary libraries
import tkinter as tk
from tkinter import messagebox
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

# Load the dataset (assuming you have a CSV file named 'gold_prices.csv' with columns 'Date' and 'Price')
data = pd.read_csv('modified_gold.csv')

# Convert 'Date' column to datetime type
# data['Date'] = pd.to_datetime(data['Date'])

# Extract year from the 'Date' column and create a new column 'Year'
# data['Year'] = data['Date'].dt.year

# Splitting data into features (X) and target variable (y)
X = data[['Year']]
X.columns = ['Year']
y = data['Price']

# # Visualizing the features (Year) against the target variable (Price)
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Gold Prices')
plt.title('Gold Prices Over the Years')
plt.xlabel('Year')
plt.ylabel('Price')
plt.grid(True)
plt.legend()
plt.show()

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Function to predict gold price for a given year
def predict_gold_price(year):
    year = [[year]]
    predicted_price = model.predict(year)
    return predicted_price[0]

# # Function to handle button click event
def predict_button_click():
    year = int(entry_year.get())
    predicted_price = predict_gold_price(year)
    # rupees = predicted_price/0.012;
    # result_label.config(text=f"Predicted gold price for {year}: ${predicted_price:.2f}")
    result_label.config(text=f"Predicted gold price for {year}: ₹{predicted_price:.2f}")



# **********************************************************************************************************************
# Creating UI
# root = tk.Tk()
# root.title("Gold Price Prediction")

# # Header
# header_label = tk.Label(root, text="Gold Price Prediction", font=("Arial", 24, "bold"))
# header_label.pack(pady=20)

# # Gold image
# gold_image = Image.open("gold.jpeg")  # Replace "gold.jpg" with the filename of your gold image
# # gold_image = gold_image.resize((200, 200), Image.ANTIALIAS)
# gold_photo = ImageTk.PhotoImage(gold_image)
# gold_label = tk.Label(root, image=gold_photo)
# gold_label.pack()

# # Year entry
# label_year = tk.Label(root, text="Enter the year:", font=("Arial", 14))
# label_year.pack(pady=10)

# entry_year = tk.Entry(root, font=("Arial", 14))
# entry_year.pack()

# # Predict button
# predict_button = tk.Button(root, text="Predict", command=predict_button_click, bg="#4caf50", fg="white", font=("Arial", 14))
# predict_button.pack(pady=10)

# # Result label
# result_label = tk.Label(root, text="", font=("Arial", 14, "bold"), fg="green")
# result_label.pack(pady=10)

# root.mainloop()

# *******************************************************************************************************************
# Creating UI
root = tk.Tk()
root.title("Gold Price Prediction")

# Header
header_label = tk.Label(root, text="Gold Price Prediction", font=("Arial", 24, "bold"))
header_label.pack(pady=20)

# Gold image
gold_image = Image.open("gold.jpeg")  # Replace "gold.jpeg" with the filename of your gold image
gold_photo = ImageTk.PhotoImage(gold_image)
gold_label = tk.Label(root, image=gold_photo)
gold_label.pack()

# Year entry
label_year = tk.Label(root, text="Enter the year:", font=("Arial", 14))
label_year.pack(pady=10)

entry_year = tk.Entry(root, font=("Arial", 14), bd=2, relief="groove", highlightbackground="#cccccc", highlightthickness=2)
entry_year.pack()


# Predict button
predict_button = tk.Button(root, text="Predict", command=predict_button_click, bg="#4caf50", fg="white", font=("Arial", 14))
predict_button.pack(pady=10)

# Result label
result_label = tk.Label(root, text="", font=("Arial", 14, "bold"), fg="green")
result_label.pack(pady=10)

root.mainloop()



