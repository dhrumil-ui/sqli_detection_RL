from flask import Blueprint

sqli_detection = Blueprint('sqli_detection', __name__)

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.svm import SVC


# Load and preprocess data
df1 = pd.read_csv("static/datasets/sqli.csv", encoding='utf-16')
df2 = pd.read_csv("static/datasets/sqliv2.csv", encoding='utf-16')
df3 = pd.read_csv("static/datasets/Modified_SQL_Dataset.csv")
df = pd.concat([df1, df2, df3]).dropna()
df['Label'] = df['Label'].astype(int)

# Preprocess data
X = df['Sentence']
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
max_sequence_length = 100
tokenizer = keras.layers.TextVectorization(max_tokens=5000, output_mode='int', output_sequence_length=max_sequence_length)
tokenizer.adapt(X_train.values)
X_train_tokens = tokenizer(X_train.values)
X_test_tokens = tokenizer(X_test.values)

# Train the CNN model
model = keras.Sequential([
    keras.layers.Embedding(input_dim=5000, output_dim=32, input_length=max_sequence_length),
    keras.layers.Conv1D(64, 5, activation='relu'),
    keras.layers.MaxPooling1D(5),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train_tokens, y_train, epochs=10, batch_size=32, validation_data=(X_test_tokens, y_test))

# Train the SVM classifier
feature_train = model.predict(X_train_tokens)
feature_test = model.predict(X_test_tokens)
svm_classifier = SVC(kernel='linear', C=1.0)
svm_classifier.fit(feature_train, y_train)

from tensorflow.keras.models import model_from_json

json_file = open('/static/cnn_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
cnn_model = model_from_json(loaded_model_json)
# load weights into new model
cnn_model.load_weights("/static/cnn_model.h5")

# # Use the loaded model to make predictions
import numpy as np
import gym
class QLearningBinaryClassifier:
    def __init__(self, num_features, learning_rate, discount_factor):
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.weights = np.zeros(num_features)

    def predict(self, state):
        q_value = np.dot(self.weights, state)
        action = 1 if q_value >= 0 else 0
        return action

    def update(self, state, action, reward, next_state):
        q_value = np.dot(self.weights, state)
        next_q_value = np.dot(self.weights, next_state)
        target = reward + self.discount_factor * next_q_value
        error = target - q_value
        self.weights += self.learning_rate * error * state
    # Define the RL agent (QLearining model )
class SQLInjectionAgent:
    def __init__(self):
        self.model = QLearningBinaryClassifier(num_features=5, learning_rate = 0.1, discount_factor = 0.9)  # Define the RL model (e.g., Q-learning, SARSA)

    def act(self, state):
        action = self.model.predict(state)  # Use the model to predict the best action
        return action
    # apriori
def is_injection(xi):
    # x_inp = X[random.randint(0,len(X)-1)]
    return cnn_model.predict(xi.reshape(-1,1,4717)).flatten()

# Define the RL environment (QLearining model )

class SQLInjectionEnv(gym.Env):

    def __init__(self):
        self.state = None
        self.done = False

    def reset(self):
        # Generate a new random SQL query to use as the initial state
        # self.state = generate_random_query()
        self.done = False
        # return self.state

    def step(self, state, action):
        # Execute the query with the given action (e.g., input, modification)
        # query = apply_action_to_query(self.state, action)
        # self.index = self.index+1
        query = state
        # query = X_train[self.index]
        result = is_injection(query)

        # Determine the reward based on the result of the query
        if is_injection(query):
            reward = -1  # Penalty for a successful injection
            self.done = True
        elif result == 'error':
            reward = -0.5  # Penalty for a failed query
        else:
            reward = 0.5  # Reward for a successful query

        return query, reward, self.done, {}
# Train the RL agent

env = SQLInjectionEnv()
agent = SQLInjectionAgent()

num_features = 4717 # featues of data
learning_rate = 0.1
discount_factor = 0.9

q_model = QLearningBinaryClassifier(num_features, learning_rate, discount_factor)

out = []

num_episodes = len(X_train)-1

for state in X_train:
    # state = env.reset() # Initialize the state
    done = False
    while not done:
        action = q_model.predict(state)
        out.append(action)
        next_state, reward, done, _ = env.step(state,action) # Observe the next state, reward, and done flag
        q_model.update(state, action, reward, next_state)
        state = next_state
half = len(X_test)//2
X_test_1 = X_test[:half]
X_val_1 = X_test[half:]
Y_test_1 = y_test[:half]
Y_val_1 = y_test[half:]
# prediction while model not learning

y_pred1 = []
for state in X_test_1:
    action = q_model.predict(state)
    y_pred1.append(action)