# house_price_prediction.py
import joblib

# Load the trained model and label encoder
loaded_model = joblib.load('house_price_model.joblib')
loaded_label_encoder = joblib.load('label_encoder.joblib')

# Function to take user input and make predictions
def predict_price(model, label_encoder, input_features):
    user_inputs = {}
    
    # Additional prompts
    user_inputs['area'] = float(input("Enter the area of the house: "))
    user_inputs['bedrooms'] = int(input("Enter the number of bedrooms: "))
    user_inputs['bathrooms'] = int(input("Enter the number of bathrooms: "))
    user_inputs['stories'] = int(input("Enter the number of stories: "))
    user_inputs['mainroad'] = input("Is it on the main road? (yes/no): ").lower()
    user_inputs['guestroom'] = input("Does it have a guest room? (yes/no): ").lower()
    user_inputs['basement'] = input("Does it have a basement? (yes/no): ").lower()
    user_inputs['hotwaterheating'] = input("Does it have hot water heating? (yes/no): ").lower()
    user_inputs['airconditioning'] = input("Does it have air conditioning? (yes/no): ").lower()
    user_inputs['parking'] = input("Does it have a parking space? (yes/no): ").lower()
    user_inputs['prefarea'] = input("Is it in a preferred area? (yes/no): ").lower()
    user_inputs['furnishingstatus'] = input("Enter furnishing status (furnished/unfurnished): ").lower()

    # Convert categorical inputs to numerical using Label Encoding
    for feature in input_features:
        user_inputs[feature] = label_encoder.transform([user_inputs[feature]])[0] if user_inputs[feature] in label_encoder.classes_ else 0  # Use 0 as a default value

    # Make a prediction using the trained model
    input_data = [user_inputs[feature] for feature in input_features]
    predicted_price = model.predict([input_data])

    print(f"\nEstimated Price for the given parameters: ${predicted_price[0]:,.2f}")

# Get the column names used during model training
input_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']

# Call the function to make predictions based on user input
predict_price(loaded_model, loaded_label_encoder, input_features)
