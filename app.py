# house_price_prediction_streamlit.py
import streamlit as st
import joblib

# Load the trained model and label encoder
loaded_model = joblib.load('house_price_model.joblib')
loaded_label_encoder = joblib.load('label_encoder.joblib')

# Function to make predictions
def predict_price(model, label_encoder, user_inputs):
    # Convert categorical inputs to numerical using Label Encoding
    for feature in user_inputs:
        user_inputs[feature] = label_encoder.transform([user_inputs[feature]])[0] if user_inputs[feature] in label_encoder.classes_ else 0  # Use 0 as a default value

    # Make a prediction using the trained model
    input_data = [user_inputs[feature] for feature in user_inputs]
    predicted_price = model.predict([input_data])

    return predicted_price[0]

# Streamlit UI
st.title("House Price Prediction")

# Collect user input using sliders and checkboxes
user_inputs = {
    'area': st.slider("Enter the area of the house:", min_value=500, max_value=5000, step=100),
    'bedrooms': st.slider("Enter the number of bedrooms:", min_value=1, max_value=10, step=1),
    'bathrooms': st.slider("Enter the number of bathrooms:", min_value=1, max_value=5, step=1),
    'stories': st.slider("Enter the number of stories:", min_value=1, max_value=4, step=1),
    'mainroad': st.checkbox("Is it on the main road?"),
    'guestroom': st.checkbox("Does it have a guest room?"),
    'basement': st.checkbox("Does it have a basement?"),
    'hotwaterheating': st.checkbox("Does it have hot water heating?"),
    'airconditioning': st.checkbox("Does it have air conditioning?"),
    'parking': st.checkbox("Does it have a parking space?"),
    'prefarea': st.checkbox("Is it in a preferred area?"),
    'furnishingstatus': st.radio("Select furnishing status:", ['furnished', 'unfurnished'])
}

# Make predictions when the user clicks the "Predict" button
if st.button("Predict"):
    predicted_price = predict_price(loaded_model, loaded_label_encoder, user_inputs)
    st.success(f"Estimated Price for the given parameters: ${predicted_price:,.2f}")
