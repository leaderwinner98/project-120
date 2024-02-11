import numpy as np
import random
import json
import pickle
import tensorflow as tf
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Words to be ignored/omitted while framing the dataset
ignore_words = ['?', '!', ',', '.', "'s", "'m"]

# Load the model
model = tf.keras.models.load_model('./chatbot_model.h5')

# Load data files
intents = json.loads(open('./intents.json').read())
words = pickle.load(open('./words.pkl', 'rb'))
classes = pickle.load(open('./classes.pkl', 'rb'))


def preprocess_user_input(user_input):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(user_input)
    stemmed_words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalnum() and word.lower() not in ignore_words]
    bag = [1 if word in stemmed_words else 0 for word in words]
    return np.array(bag)


def bot_class_prediction(user_input):
    inp = preprocess_user_input(user_input)
    prediction = model.predict(np.array([inp]))
    predicted_class_label = np.argmax(prediction[0])
    return predicted_class_label


def bot_response(user_input):
    predicted_class_label = bot_class_prediction(user_input)

    # Extract the class from the predicted_class_label
    predicted_class = classes[predicted_class_label]

    # Now we have the predicted tag, select a random response
    for intent in intents['intents']:
        if intent['tag'] == predicted_class:
            responses = intent['responses']
            bot_response = random.choice(responses)
            return bot_response


print("Hi, I am Stella. How can I help you?")

while True:
    # Take input from the user
    user_input = input('Type your message here: ')

    response = bot_response(user_input)
    print("Bot Response:", response)
