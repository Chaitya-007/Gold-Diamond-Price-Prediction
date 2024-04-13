# Importing necessary libraries
import tkinter as tk
from tkinter import messagebox
from datetime import datetime
from tkinter.font import BOLD
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from PIL import Image, ImageTk, ImageOps
import joblib  # For loading the SVM model
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR


svm_model = joblib.load("svm_gold_model.pkl")
svm_model_diamond = joblib.load("svm_diamond_model.pkl")


#! Adding petrol dataset
# Load the dataset (assuming you have a CSV file named 'petrol_prices.csv' with columns 'Year' and 'Price')
data = pd.read_csv('petrol_prices.csv')

# Splitting data into features (X) and target variable (y)
X_petrol = data[['Year']]
X_petrol.columns = ['Year']
y_petrol = data['Price']


# Splitting the gold dataset into training and testing sets
X_train_petrol, X_test_petrol, y_train_petrol, y_test_petrol = train_test_split(X_petrol, y_petrol, test_size=0.2, random_state=42)

# Training the linear regression model for gold
model_petrol = LinearRegression()
model_petrol.fit(X_train_petrol, y_train_petrol)

# Training the decision tree regression model for gold
model_decision_tree_petrol = DecisionTreeRegressor(random_state=42)
model_decision_tree_petrol.fit(X_train_petrol, y_train_petrol)

# Training the KNN regression model for gold
knn_model_petrol = KNeighborsRegressor(n_neighbors=5)
knn_model_petrol.fit(X_train_petrol, y_train_petrol)


def display_petrol_analysis_graph():
    # Visualizing the gold prices over the years
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='Green', label='Petrol Prices')
    plt.title('Petrol Prices Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Price')
    plt.grid(True)
    plt.legend()
    plt.show()

# !*****************************************************************************************************************************

#! Adding diesel dataset
# Load the dataset (assuming you have a CSV file named 'diesel_prices.csv' with columns 'Year' and 'Price')
data = pd.read_csv('diesel_prices.csv')

# Splitting data into features (X) and target variable (y)
X_diesel = data[['Year']]
X_diesel.columns = ['Year']
y_diesel = data['Price']


# Splitting the gold dataset into training and testing sets
X_train_diesel, X_test_diesel, y_train_diesel, y_test_diesel = train_test_split(X_diesel, y_diesel, test_size=0.2, random_state=42)

# Training the linear regression model for gold
model_diesel = LinearRegression()
model_diesel.fit(X_train_diesel, y_train_diesel)

# Training the decision tree regression model for gold
model_decision_tree_diesel = DecisionTreeRegressor(random_state=42)
model_decision_tree_diesel.fit(X_train_diesel, y_train_diesel)

# Training the KNN regression model for gold
knn_model_diesel = KNeighborsRegressor(n_neighbors=5)
knn_model_diesel.fit(X_train_diesel, y_train_diesel)


def display_diesel_analysis_graph():
    # Visualizing the gold prices over the years
    plt.figure(figsize=(10, 6))
    plt.scatter(X_diesel, y_diesel, color='yellow', label='Diesel Prices')
    plt.title('Diesel Prices Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Price')
    plt.grid(True)
    plt.legend()
    plt.show()

# !*****************************************************************************************************************************

# Load the dataset (assuming you have a CSV file named 'modified_gold.csv' with columns 'Year' and 'Price')
data = pd.read_csv('modified_gold.csv')

# Splitting data into features (X) and target variable (y)
X = data[['Year']]
X.columns = ['Year']
y = data['Price']


# Splitting the gold dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the linear regression model for gold
model = LinearRegression()
model.fit(X_train, y_train)

# Training the decision tree regression model for gold
model_decision_tree = DecisionTreeRegressor(random_state=42)
model_decision_tree.fit(X_train, y_train)

# Training the KNN regression model for gold
knn_model_gold = KNeighborsRegressor(n_neighbors=5)
knn_model_gold.fit(X_train, y_train)


def display_gold_analysis_graph():
    # Visualizing the gold prices over the years
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Gold Prices')
    plt.title('Gold Prices Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Price')
    plt.grid(True)
    plt.legend()
    plt.show()
    
# !*********************************************************************************************************************************************

# Load the dataset for diamonds (assuming you have a CSV file named 'diamond_prices.csv' with columns 'Year' and 'Price')
diamond_data = pd.read_csv('diamond_prices.csv')


# Splitting data into features (X_diamond) and target variable (y_diamond)
X_diamond = diamond_data[['Year']]
X_diamond.columns = ['Year']
y_diamond = diamond_data['Price']


# Splitting the diamond dataset into training and testing sets
X_train_diamond, X_test_diamond, y_train_diamond, y_test_diamond = train_test_split(X_diamond, y_diamond, test_size=0.2, random_state=42)

# Training the linear regression model for diamonds
model_diamond = LinearRegression()
model_diamond.fit(X_train_diamond, y_train_diamond)

# Training the KNN regression model for diamonds
knn_model_diamond = KNeighborsRegressor(n_neighbors=5)
knn_model_diamond.fit(X_train_diamond, y_train_diamond)

model_decision_tree_diamond = DecisionTreeRegressor(random_state=42)
model_decision_tree_diamond.fit(X_train_diamond, y_train_diamond)

def display_diamond_analysis_graph():
    # Visualizing the diamond prices over the years
    plt.figure(figsize=(10, 6))
    plt.scatter(X_diamond, y_diamond, color='red', label='Diamond Prices')
    plt.title('Diamond Prices Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Price')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    
# !********************************************************************************************************************************************* 
    
def calculate_accuracy(model, X_test, y_test):
    accuracy = model.score(X_test, y_test)
    return accuracy

def calculate_accuracy_svm(model, X_test, y_test):
    accuracy = model.score(X_test, y_test)
    return accuracy


# def display_accuracy():
#     gold_linear_accuracy = calculate_accuracy(model, X_test, y_test)
#     gold_decision_tree_accuracy = calculate_accuracy(model_decision_tree, X_test, y_test)
#     gold_knn_accuracy = calculate_accuracy(knn_model_gold, X_test, y_test)
#     diamond_linear_accuracy = calculate_accuracy(model_diamond, X_test_diamond, y_test_diamond)
#     diamond_decision_tree_accuracy = calculate_accuracy(model_decision_tree, X_test_diamond, y_test_diamond)
#     diamond_knn_accuracy = calculate_accuracy(knn_model_diamond, X_test_diamond, y_test_diamond)

#     messagebox.showinfo("Accuracy Scores", 
#                         f"Gold Linear Regression Accuracy: {gold_linear_accuracy}\n"
#                         f"Gold Decision Tree Accuracy: {gold_decision_tree_accuracy}\n"
#                         f"Gold KNN Accuracy: {gold_knn_accuracy}\n"
#                         f"Diamond Linear Regression Accuracy: {diamond_linear_accuracy}\n"
#                         f"Diamond Decision Tree Accuracy: {diamond_decision_tree_accuracy}\n"
#                         f"Diamond KNN Accuracy: {diamond_knn_accuracy}\n")

def open_accuracy_window():
    accuracy_window = tk.Toplevel(root)
    accuracy_window.title("Accuracy Scores")

    # display_accuracy_button = tk.Button(accuracy_window, text="Display Accuracy", command=display_accuracy, bg="#4caf50", fg="white", font=("Arial", 12, "bold"))
    # display_accuracy_button.pack(pady=10)
    gold_linear_accuracy = calculate_accuracy(model, X_test, y_test)
    gold_decision_tree_accuracy = calculate_accuracy(model_decision_tree, X_test, y_test)
    gold_knn_accuracy = calculate_accuracy(knn_model_gold, X_test, y_test)
    gold_svm_accuracy = calculate_accuracy_svm(svm_model, X_test, y_test)  # Assuming svm_model_gold is defined
    diamond_linear_accuracy = calculate_accuracy(model_diamond, X_test_diamond, y_test_diamond)
    diamond_decision_tree_accuracy = calculate_accuracy(model_decision_tree, X_test_diamond, y_test_diamond)
    diamond_knn_accuracy = calculate_accuracy(knn_model_diamond, X_test_diamond, y_test_diamond)
    diamond_svm_accuracy = calculate_accuracy_svm(svm_model_diamond, X_test_diamond, y_test_diamond)

    messagebox.showinfo("Accuracy Scores", 
                        f"Gold Linear Regression Accuracy: {gold_linear_accuracy}\n"
                        f"Gold Decision Tree Accuracy: {gold_decision_tree_accuracy}\n"
                        f"Gold KNN Accuracy: {gold_knn_accuracy}\n"
                        f"Gold SVM Accuracy: {gold_svm_accuracy}\n"
                        f"Diamond Linear Regression Accuracy: {diamond_linear_accuracy}\n"
                        f"Diamond Decision Tree Accuracy: {diamond_decision_tree_accuracy}\n"
                        f"Diamond KNN Accuracy: {diamond_knn_accuracy}\n")
                        # f"Diamond SVM Accuracy: {diamond_svm_accuracy}\n")





# def proceed():
#     selected_prediction = var.get()
#     if selected_prediction == "Gold":
#         open_gold_prediction_window()
#     elif selected_prediction == "Diamond":
#         open_diamond_prediction_window()

def proceed():
    selected_prediction = var.get()
    if selected_prediction == "Gold":
        open_gold_prediction_window()
    elif selected_prediction == "Diamond":
        open_diamond_prediction_window()
    elif selected_prediction == "Petrol":
        open_petrol_prediction_window()
    elif selected_prediction == "Diesel":
        open_diesel_prediction_window()


def open_gold_prediction_window():
    
    def predict_gold_price(year):
        year = [[year]]
        predicted_price = model.predict(year)
        return predicted_price[0]
    
    def predict_gold_price_svm(year):
        try:
            # Load the SVM model from file
            svm_model = joblib.load("svm_gold_model.pkl")
            year = [[year]]
            predicted_price = svm_model.predict(year)
            return predicted_price[0]
        except Exception as e:
            print("Error:", e)
            return None
        
    def predict_gold_price_decision_tree(year):
        year = [[year]]
        predicted_price = model_decision_tree.predict(year)
        return predicted_price[0]
    
    def predict_gold_price_knn(year):
        year = [[year]]
        predicted_price = knn_model_gold.predict(year)
        return predicted_price[0]
    
    def predict_gold_button_click():
        try:
            year = int(gold_entry.get())
            predicted_price = predict_gold_price(year)  # Assuming predict_gold_price function is defined elsewhere
            # result_label.config(text=f"Predicted Gold Price: ₹{predicted_price:.2f}", fg="green")
            predicted_price_svm = predict_gold_price_svm(year)
            predicted_price_decision_tree = predict_gold_price_decision_tree(year)
            predicted_price_knn = predict_gold_price_knn(year)
            result_label.config(text=f"Predicted Gold Price (Linear Regression): ₹{predicted_price:.2f}\n"f"Predicted Gold Price (SVM): ₹{predicted_price_svm:.2f}\n"f"Predicted Gold Price (Decision Tree): ₹{predicted_price_decision_tree:.2f}\n"f"Predicted Gold Price (KNN): ₹{predicted_price_knn:.2f}\n", fg="green")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid year.")
            
    def on_closing():
        gold_window.destroy()
        display_gold_analysis_graph()
            
    gold_window = tk.Toplevel(root)
    gold_window.title("Gold Price Prediction")
    gold_window.protocol("WM_DELETE_WINDOW", on_closing)
    
    gold_label = tk.Label(gold_window, text="Gold Price Prediction", font=("Arial", 16, "bold"))
    gold_label.pack(pady=10)
    
    gold_image = Image.open("gold.jpeg")
    # gold_image = gold_image.resize((200, 200), Image.ANTIALIAS)
    gold_photo = ImageTk.PhotoImage(gold_image)
    gold_image_label = tk.Label(gold_window, image=gold_photo)
    gold_image_label.image = gold_photo
    gold_image_label.pack()
    
    gold_question_label = tk.Label(gold_window, text="Enter the year for which the price needs to be predicted:", font=("Arial", 12, "bold"))
    gold_question_label.pack(pady=10)
    
    gold_entry = tk.Entry(gold_window, font=("Arial", 12), bd=2, relief="groove", highlightbackground="#cccccc", highlightthickness=2)
    gold_entry.pack()
    
    gold_button = tk.Button(gold_window, text="Predict Price", command=predict_gold_button_click, bg="#4caf50", fg="white", font=("Arial", 12, "bold"))
    gold_button.pack(pady=10)
    
    result_label = tk.Label(gold_window, text="", font=("Arial", 12, "bold"))
    result_label.pack(pady=10)
    
    
# ***************************************************************************************************

def open_diamond_prediction_window():
    
    def predict_diamond_price(year):
        year = [[year]]
        predicted_price = model_diamond.predict(year)
        return predicted_price[0]
    
    def predict_diamond_price_svm(year):
        try:
            # Load the SVM model from file
            svm_model = joblib.load("svm_diamond_model.pkl")
            year = [[year]]
            predicted_price = svm_model.predict(year)
            return predicted_price[0]
        except Exception as e:
            print("Error:", e)
            return None
        
    def predict_diamond_price_decision_tree(year):
        year = [[year]]
        predicted_price = model_decision_tree_diamond.predict(year)
        return predicted_price[0]
    
    def predict_diamond_price_knn(year):
        year = [[year]]
        predicted_price = knn_model_diamond.predict(year)
        return predicted_price[0]

    
    def predict_diamond_button_click():
        try:
            year = int(diamond_entry.get())
            predicted_price = predict_diamond_price(year)  # Assuming predict_diamond_price function is defined elsewhere
            predicted_price_svm = predict_diamond_price_svm(year)
            # result_label.config(text=f"Predicted Diamond Price: ₹{predicted_price:.2f}", fg="green")
            predicted_price_decision_tree = predict_diamond_price_decision_tree(year)
            predicted_price_knn = predict_diamond_price_knn(year)
            result_label.config(text=f"Predicted Diamond Price (Linear Regression): ₹{predicted_price:.2f}\n"f"Predicted Diamond Price (SVM): ₹{predicted_price_svm:.2f}\n"f"Predicted Diamond Price (Decision Tree): ₹{predicted_price_decision_tree:.2f}\n"f"Predicted Diamond Price (KNN)): ₹{predicted_price_knn:.2f}", fg="green")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid year.")
            
    def on_closing():
        diamond_window.destroy()
        display_diamond_analysis_graph()
            
    diamond_window = tk.Toplevel(root)
    diamond_window.title("Diamond Price Prediction")
    diamond_window.protocol("WM_DELETE_WINDOW", on_closing)
    
    diamond_label = tk.Label(diamond_window, text="Diamond Price Prediction", font=("Arial", 16, "bold"))
    diamond_label.pack(pady=10)
    
    diamond_image = Image.open("diamond.jpeg")
    # diamond_image = diamond_image.resize((200, 200), Image.ANTIALIAS)
    diamond_photo = ImageTk.PhotoImage(diamond_image)
    diamond_image_label = tk.Label(diamond_window, image=diamond_photo)
    diamond_image_label.image = diamond_photo
    diamond_image_label.pack()
    
    diamond_question_label = tk.Label(diamond_window, text="Enter the year for which the price needs to be predicted:", font=("Arial", 12, "bold"))
    diamond_question_label.pack(pady=10)
    
    diamond_entry = tk.Entry(diamond_window, font=("Arial", 12), bd=2, relief="groove", highlightbackground="#cccccc", highlightthickness=2)
    diamond_entry.pack()
    
    diamond_button = tk.Button(diamond_window, text="Predict Price", command=predict_diamond_button_click, bg="#4caf50", fg="white", font=("Arial", 12, "bold"))
    diamond_button.pack(pady=10)
    
    result_label = tk.Label(diamond_window, text="", font=("Arial", 12))
    result_label.pack(pady=10)



# ***************************************************************************************************

def open_petrol_prediction_window():
    
    def predict_petrol_price(year):
        year = [[year]]
        predicted_price = model_diamond.predict(year)
        return predicted_price[0]
    
    def predict_petrol_price_svm(year):
        try:
            # Load the SVM model from file
            svm_model = joblib.load("svm_petrol_model.pkl")
            year = [[year]]
            predicted_price = svm_model.predict(year)
            return predicted_price[0]
        except Exception as e:
            print("Error:", e)
            return None
        
    def predict_petrol_price_decision_tree(year):
        year = [[year]]
        predicted_price = model_decision_tree_petrol.predict(year)
        return predicted_price[0]
    
    def predict_petrol_price_knn(year):
        year = [[year]]
        predicted_price = knn_model_petrol.predict(year)
        return predicted_price[0]

    
    def predict_petrol_button_click():
        try:
            year = int(petrol_entry.get())
            predicted_price = predict_petrol_price(year)  # Assuming predict_diamond_price function is defined elsewhere
            predicted_price_svm = predict_petrol_price_svm(year)
            # result_label.config(text=f"Predicted Diamond Price: ₹{predicted_price:.2f}", fg="green")
            predicted_price_decision_tree = predict_petrol_price_decision_tree(year)
            predicted_price_knn = predict_petrol_price_knn(year)
            result_label.config(text=f"Predicted Petrol Price (Linear Regression): ₹{predicted_price:.2f}\n"f"Predicted Petrol Price (SVM): ₹{predicted_price_svm:.2f}\n"f"Predicted Petrol Price (Decision Tree): ₹{predicted_price_decision_tree:.2f}\n"f"Predicted Petrol Price (KNN)): ₹{predicted_price_knn:.2f}", fg="green")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid year.")
            
    def on_closing():
        petrol_window.destroy()
        display_petrol_analysis_graph()
            
    petrol_window = tk.Toplevel(root)
    petrol_window.title("Petrol Price Prediction")
    petrol_window.protocol("WM_DELETE_WINDOW", on_closing)
    
    petrol_label = tk.Label(petrol_window, text="Diamond Price Prediction", font=("Arial", 16, "bold"))
    petrol_label.pack(pady=10)
    
    petrol_image = Image.open("petrol.jpeg")
    petrol_image = petrol_image.resize((200, 200))
    petrol_photo = ImageTk.PhotoImage(petrol_image)
    petrol_image_label = tk.Label(petrol_window, image=petrol_photo)
    petrol_image_label.image = petrol_photo
    petrol_image_label.pack()
    
    petrol_question_label = tk.Label(petrol_window, text="Enter the year for which the price needs to be predicted:", font=("Arial", 12, "bold"))
    petrol_question_label.pack(pady=10)
    
    petrol_entry = tk.Entry(petrol_window, font=("Arial", 12), bd=2, relief="groove", highlightbackground="#cccccc", highlightthickness=2)
    petrol_entry.pack()
    
    petrol_button = tk.Button(petrol_window, text="Predict Price", command=predict_petrol_button_click, bg="#4caf50", fg="white", font=("Arial", 12, "bold"))
    petrol_button.pack(pady=10)
    
    result_label = tk.Label(petrol_window, text="", font=("Arial", 12))
    result_label.pack(pady=10)



# ***************************************************************************************************


def open_diesel_prediction_window():
    
    def predict_diesel_price(year):
        year = [[year]]
        predicted_price = model_diesel.predict(year)
        return predicted_price[0]
    
    def predict_diesel_price_svm(year):
        try:
            # Load the SVM model from file
            svm_model = joblib.load("svm_diesel_model.pkl")
            year = [[year]]
            predicted_price = svm_model.predict(year)
            return predicted_price[0]
        except Exception as e:
            print("Error:", e)
            return None
        
    def predict_diesel_price_decision_tree(year):
        year = [[year]]
        predicted_price = model_decision_tree_diesel.predict(year)
        return predicted_price[0]
    
    def predict_diesel_price_knn(year):
        year = [[year]]
        predicted_price = knn_model_diesel.predict(year)
        return predicted_price[0]

    
    def predict_diesel_button_click():
        try:
            year = int(diesel_entry.get())
            predicted_price = predict_diesel_price(year)  # Assuming predict_diamond_price function is defined elsewhere
            predicted_price_svm = predict_diesel_price_svm(year)
            # result_label.config(text=f"Predicted Diamond Price: ₹{predicted_price:.2f}", fg="green")
            predicted_price_decision_tree = predict_diesel_price_decision_tree(year)
            predicted_price_knn = predict_diesel_price_knn(year)
            result_label.config(text=f"Predicted Diesel Price (Linear Regression): ₹{predicted_price:.2f}\n"f"Predicted Diesel Price (SVM): ₹{predicted_price_svm:.2f}\n"f"Predicted Diesel Price (Decision Tree): ₹{predicted_price_decision_tree:.2f}\n"f"Predicted Diesel Price (KNN)): ₹{predicted_price_knn:.2f}", fg="green")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid year.")
            
    def on_closing():
        diesel_window.destroy()
        display_diesel_analysis_graph()
            
    diesel_window = tk.Toplevel(root)
    diesel_window.title("diesel Price Prediction")
    diesel_window.protocol("WM_DELETE_WINDOW", on_closing)
    
    diesel_label = tk.Label(diesel_window, text="Diamond Price Prediction", font=("Arial", 16, "bold"))
    diesel_label.pack(pady=10)
    
    diesel_image = Image.open("diesel.jpeg")
    diesel_image = diesel_image.resize((200, 200))
    diesel_photo = ImageTk.PhotoImage(diesel_image)
    diesel_image_label = tk.Label(diesel_window, image=diesel_photo)
    diesel_image_label.image = diesel_photo
    diesel_image_label.pack()
    
    diesel_question_label = tk.Label(diesel_window, text="Enter the year for which the price needs to be predicted:", font=("Arial", 12, "bold"))
    diesel_question_label.pack(pady=10)
    
    diesel_entry = tk.Entry(diesel_window, font=("Arial", 12), bd=2, relief="groove", highlightbackground="#cccccc", highlightthickness=2)
    diesel_entry.pack()
    
    diesel_button = tk.Button(diesel_window, text="Predict Price", command=predict_diesel_button_click, bg="#4caf50", fg="white", font=("Arial", 12, "bold"))
    diesel_button.pack(pady=10)
    
    result_label = tk.Label(diesel_window, text="", font=("Arial", 12))
    result_label.pack(pady=10)



# ***************************************************************************************************






root = tk.Tk()
root.title("Price Prediction Using Machine Learning")

# Get the width and height of the screen
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Set the geometry of the window to cover the entire screen
root.geometry(f"{screen_width}x{screen_height}")

# Title above the image
title_label = tk.Label(root, text="Gold and Diamond Price Predictor", font=("Arial", 16, "bold"))
title_label.pack(pady=10)

# Load the prediction image
prediction_image = Image.open("prediction_02.jpeg")
remaining_width = int(screen_width - (screen_width * 0.15 * 2))
image_width = remaining_width
image_height = screen_height // 2
resized_prediction_image = ImageOps.fit(prediction_image, (image_width, image_height))
resized_prediction_photo = ImageTk.PhotoImage(resized_prediction_image)
prediction_label = tk.Label(root, image=resized_prediction_photo)
prediction_label.place(x=int(screen_width * 0.15), y=50)

# Calculate the x-coordinate for centering the widget
x_center = screen_width // 2
x_start = screen_width * 0.25
x_start_second = screen_width * 0.75

question_label = tk.Label(root, text="Which price do you want to predict?", font=("Arial", 16, "bold"))
question_label.place(x=x_center, y=50 + image_height + 30, anchor="center")

# Radio button selection for prediction
var = tk.StringVar()
var.set(None)  # Set initial value to None
# gold_radio = tk.Radiobutton(root, text="Gold Price Prediction", variable=var, value="Gold", font=("Arial", 16, "bold"))
# gold_radio.place(x=x_center, y=50 + image_height + 70, anchor="center")

gold_radio = tk.Radiobutton(root, text="Gold Price Prediction ", variable=var, value="Gold", font=("Arial", 16, "bold"))
gold_radio.place(x=x_start, y=50 + image_height + 70, anchor="center")

diamond_radio = tk.Radiobutton(root, text="Diamond Price Prediction", variable=var, value="Diamond", font=("Arial", 16, "bold"))
diamond_radio.place(x=x_start_second, y=50 + image_height + 70, anchor="center")

petrol_radio = tk.Radiobutton(root, text="Petrol Price Prediction", variable=var, value="Petrol", font=("Arial", 16, "bold"))
petrol_radio.place(x=x_start, y=50 + image_height + 110, anchor="center")

diesel_radio = tk.Radiobutton(root, text="Diesel Price Prediction    ", variable=var, value="Diesel", font=("Arial", 16, "bold"))
diesel_radio.place(x=x_start_second, y=50 + image_height + 110, anchor="center")

# Proceed button
proceed_button = tk.Button(root, text="Proceed", command=proceed, bg="#4caf50", fg="white", font=("Arial", 16, "bold"))
proceed_button.place(x=x_center, y=50 + image_height + 160, anchor="center")

# Accuracy button
accuracy_button = tk.Button(root, text="Display Accuracy", command=open_accuracy_window, bg="#4caf50", fg="white", font=("Arial", 16, "bold"))
accuracy_button.place(x=x_center, y=50 + image_height + 210, anchor="center")

root.mainloop()

