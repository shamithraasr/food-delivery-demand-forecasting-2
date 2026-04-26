import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
data = {
    'Day': list(range(1, 11)),
    'Orders': [120, 135, 150, 140, 165, 180, 175, 190, 210, 220]
}

df = pd.DataFrame(data)
df['Moving_Avg'] = df['Orders'].rolling(window=3).mean()
X = df[['Day']]
y = df['Orders']

model = LinearRegression()
model.fit(X, y)
next_day = np.array([[11]])
prediction = model.predict(next_day)

# -----------------------------
# Plot 1: Original Demand
# -----------------------------
plt.figure()
plt.plot(df['Day'], df['Orders'], marker='o')
plt.title("Food Delivery Demand - Original Data")
plt.xlabel("Day")
plt.ylabel("Orders")
plt.grid()
plt.show()

# -----------------------------
# Plot 2: Moving Average
# -----------------------------
plt.figure()
plt.plot(df['Day'], df['Moving_Avg'], marker='o')
plt.title("3-Day Moving Average (Smoothed Demand)")
plt.xlabel("Day")
plt.ylabel("Orders")
plt.grid()
plt.show()

# -----------------------------
# Plot 3: Forecast
# -----------------------------
plt.figure()
plt.plot(df['Day'], df['Orders'], marker='o', label="Actual Demand")
plt.scatter(11, prediction, label="Predicted Next Day")
plt.title("Demand Forecast")
plt.xlabel("Day")
plt.ylabel("Orders")
plt.legend()
plt.grid()
plt.show()

print("Predicted Orders for Next Day:", round(prediction[0], 2))

