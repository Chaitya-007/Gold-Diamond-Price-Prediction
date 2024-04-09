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
# def predict_gold_price(year):
#     year = [[year]]
#     predicted_price = model.predict(year)
#     return predicted_price[0]

# Function to handle button click event for gold prediction
# def predict_gold_button_click():
#     year = int(entry_year_gold.get())
#     predicted_price = predict_gold_price(year)
#     result_label_gold.config(text=f"Predicted gold price for {year}: ₹{predicted_price:.2f}")


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


def proceed():
    selected_prediction = var.get()
    if selected_prediction == "Gold":
        open_gold_prediction_window()
    elif selected_prediction == "Diamond":
        open_diamond_prediction_window()

def open_gold_prediction_window():
    
    def predict_gold_price(year):
        year = [[year]]
        predicted_price = model.predict(year)
        return predicted_price[0]
    
    def predict_gold_button_click():
        try:
            year = int(gold_entry.get())
            predicted_price = predict_gold_price(year)  # Assuming predict_gold_price function is defined elsewhere
            result_label.config(text=f"Predicted Gold Price: ₹{predicted_price:.2f}", fg="green")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid year.")
            
    gold_window = tk.Toplevel(root)
    gold_window.title("Gold Price Prediction")
    
    gold_label = tk.Label(gold_window, text="Gold Price Prediction", font=("Arial", 16, "bold"))
    gold_label.pack(pady=10)
    
    gold_image = Image.open("gold.jpeg")
    # gold_image = gold_image.resize((200, 200), Image.ANTIALIAS)
    gold_photo = ImageTk.PhotoImage(gold_image)
    gold_image_label = tk.Label(gold_window, image=gold_photo)
    gold_image_label.image = gold_photo
    gold_image_label.pack()
    
    gold_question_label = tk.Label(gold_window, text="Enter the year for which the price needs to be predicted:", font=("Arial", 12))
    gold_question_label.pack(pady=10)
    
    gold_entry = tk.Entry(gold_window, font=("Arial", 12), bd=2, relief="groove", highlightbackground="#cccccc", highlightthickness=2)
    gold_entry.pack()
    
    gold_button = tk.Button(gold_window, text="Predict Price", command=predict_gold_button_click, bg="#4caf50", fg="white", font=("Arial", 12))
    gold_button.pack(pady=10)
    
    result_label = tk.Label(gold_window, text="", font=("Arial", 12))
    result_label.pack(pady=10)

def open_diamond_prediction_window():
    
    def predict_diamond_price(year):
        year = [[year]]
        predicted_price = model_diamond.predict(year)
        return predicted_price[0]
    
    def predict_diamond_button_click():
        try:
            year = int(diamond_entry.get())
            predicted_price = predict_diamond_price(year)  # Assuming predict_diamond_price function is defined elsewhere
            result_label.config(text=f"Predicted Diamond Price: ₹{predicted_price:.2f}", fg="green")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid year.")
            
    diamond_window = tk.Toplevel(root)
    diamond_window.title("Diamond Price Prediction")
    
    diamond_label = tk.Label(diamond_window, text="Diamond Price Prediction", font=("Arial", 16, "bold"))
    diamond_label.pack(pady=10)
    
    diamond_image = Image.open("diamond.jpeg")
    # diamond_image = diamond_image.resize((200, 200), Image.ANTIALIAS)
    diamond_photo = ImageTk.PhotoImage(diamond_image)
    diamond_image_label = tk.Label(diamond_window, image=diamond_photo)
    diamond_image_label.image = diamond_photo
    diamond_image_label.pack()
    
    diamond_question_label = tk.Label(diamond_window, text="Enter the year for which the price needs to be predicted:", font=("Arial", 12))
    diamond_question_label.pack(pady=10)
    
    diamond_entry = tk.Entry(diamond_window, font=("Arial", 12), bd=2, relief="groove", highlightbackground="#cccccc", highlightthickness=2)
    diamond_entry.pack()
    
    diamond_button = tk.Button(diamond_window, text="Predict Price", command=predict_diamond_button_click, bg="#4caf50", fg="white", font=("Arial", 12))
    diamond_button.pack(pady=10)
    
    result_label = tk.Label(diamond_window, text="", font=("Arial", 12))
    result_label.pack(pady=10)

# Creating main window
root = tk.Tk()
root.title("Price Prediction Using Machine Learning")
# root.geometry("400x600")
# Get the width and height of the screen
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Set the geometry of the window to cover the entire screen
root.geometry(f"{screen_width}x{screen_height}")

# Title above the image
title_label = tk.Label(root, text="Gold and Diamond Price Predictor", font=("Arial", 16, "bold"))
title_label.pack(pady=10)


# Load and display prediction image
prediction_image = Image.open("prediction_02.jpeg")
prediction_photo = ImageTk.PhotoImage(prediction_image)
prediction_label = tk.Label(root, image=prediction_photo)
prediction_label.pack(pady=10)

# Question label
question_label = tk.Label(root, text="Which price do you want to predict?", font=("Arial", 14))
question_label.pack(pady=10)

# Radio button selection for prediction
var = tk.StringVar()
# var.set("") 
var.set(None)  # Set initial value to None

gold_radio = tk.Radiobutton(root, text="Gold Price Prediction", variable=var, value="Gold", font=("Arial", 12))
gold_radio.pack()

diamond_radio = tk.Radiobutton(root, text="Diamond Price Prediction", variable=var, value="Diamond", font=("Arial", 12))
diamond_radio.pack()

# Proceed button
proceed_button = tk.Button(root, text="Proceed", command=proceed, bg="#4caf50", fg="white", font=("Arial", 12))
proceed_button.pack(pady=10)

root.mainloop()
