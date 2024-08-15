import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load the California Housing dataset
df = pd.read_csv('C:/Users/senth/Downloads/housing.csv/housing.csv')

# Preprocess the data
# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='median')
df_imputed = imputer.fit_transform(df)

# Convert the imputed data back to a Pandas DataFrame
df_imputed = pd.DataFrame(df_imputed, columns=df.columns)

# Select all features except 'median_house_value'
X = df_imputed[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']]
y = df_imputed['median_house_value']

# Scale numerical features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert the scaled data back to a Pandas DataFrame
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Evaluate the model using Mean Squared Error (MSE)
y_pred_lr = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)

print('Linear Regression Model:')
print(f'Mean Squared Error (MSE): {mse_lr:.2f}')

# Ask for inputs from the console for the new property
print('Enter the details of the new property:')
longitude = float(input('Longitude: '))
latitude = float(input('Latitude: '))
housing_median_age = int(input('Housing Median Age: '))
total_rooms = int(input('Total Rooms: '))
total_bedrooms = int(input('Total Bedrooms: '))
population = int(input('Population: '))
households = int(input('Households: '))
median_income = int(input('Median Income: '))

# Create a new property DataFrame with 8 features
new_property = pd.DataFrame({
    'longitude': [longitude],
    'latitude': [latitude],
    'housing_median_age': [housing_median_age],
    'total_rooms': [total_rooms],
    'total_bedrooms': [total_bedrooms],
    'population': [population],
    'households': [households],
    'median_income': [median_income]
})

# Scale the new property data
new_property_scaled = scaler.transform(new_property)

# Use the Linear Regression model to make predictions on the new property
predicted_price = lr_model.predict(new_property_scaled)[0]

print(f'Predicted House Price: ${predicted_price:.2f}')
