from flask import Flask, render_template,request,session
from flask_sqlalchemy import SQLAlchemy
# from routes import sqli_detection
import pandas as pd
from sklearn.metrics import accuracy_score
# import tensorflow as tf
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.svm import SVC
from datetime import datetime, timedelta
import time


df1 = pd.read_csv("static/datasets/sqli.csv", encoding='utf-16')
df2 = pd.read_csv("static/datasets/sqliv2.csv", encoding='utf-16')
df3 = pd.read_csv("static/datasets/Modified_SQL_Dataset.csv")
df = pd.concat([df1, df2, df3]).dropna()
df['Label'] = df['Label'].astype(int)


X = df['Sentence']
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
max_sequence_length = 100
tokenizer = keras.layers.TextVectorization(max_tokens=5000, output_mode='int', output_sequence_length=max_sequence_length)
tokenizer.adapt(X_train.values)
X_train_tokens = tokenizer(X_train.values)
X_test_tokens = tokenizer(X_test.values)

model = keras.Sequential([
    keras.layers.Embedding(input_dim=5000, output_dim=32, input_length=max_sequence_length),
    keras.layers.Conv1D(64, 5, activation='relu', padding='same'),
    keras.layers.MaxPooling1D(5),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train_tokens, y_train, epochs=10, batch_size=32, validation_data=(X_test_tokens, y_test))
feature_test = model.predict(X_test_tokens)
feature_train = model.predict(X_train_tokens)
y_pred_binary = (feature_test > 0.5).astype(int)


svm_classifier = SVC(kernel='linear', C=1.0)
svm_classifier.fit(feature_train, y_train)
y_pred = svm_classifier.predict(feature_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred) * 100
# count=0
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost:3307/major'
db = SQLAlchemy(app)
app.secret_key = 'super-secret-key' 

class Log_in(db.Model):       
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), nullable=False)
    password = db.Column(db.String(10), nullable=False)

failed_login_attempts = {}
disabled_users = {}

@app.route("/")
def Login():
    return render_template('login.html')

@app.route("/index", methods=['GET', 'POST'])
def UI():
    global failed_login_attempts
    
    if request.method == 'POST':
        new_query = request.form.get('username')
        password = request.form.get('password')
        
        # Check if the user has exceeded the login attempts or SQL injection detected
        if new_query in failed_login_attempts and failed_login_attempts[new_query] >= 3 and datetime.now() < disabled_users[new_query]:
            remaining_time = int((disabled_users[new_query] - datetime.now()).total_seconds())
            return render_template('login.html', msg=f'Login disabled. Please try again in {remaining_time} seconds.')

        if new_query is None or len(new_query) == 0:
            result = "Please provide a valid username."
            return render_template('login.html',msg=result)
        else:
            new_query_tokens = tokenizer([new_query])  
            new_query_feature = model.predict(new_query_tokens)  
            new_query_pred = svm_classifier.predict(new_query_feature)  
            if new_query_pred[0] == 1:
                result = "SQL Injection detected!"
                # Disable login button for 20 seconds
                failed_login_attempts[new_query] = failed_login_attempts.get(new_query, 0) + 1
                disabled_users[new_query] = datetime.now() + timedelta(seconds=20)
                return render_template('index.html', query=new_query, res=result, acc= accuracy)
            else:
                result = "SQL Injection not detected."
                account = Log_in.query.filter_by(username=new_query, password=password).first()
                if account:
                    # Reset failed login attempts
                    failed_login_attempts.pop(new_query, None)
                    return render_template('home.html', username=new_query)
                else:
                    # Increment failed login attempts
                    failed_login_attempts[new_query] = failed_login_attempts.get(new_query, 0) + 1
                    if failed_login_attempts[new_query] >= 3:
                        # Disable login for 20 seconds
                        disabled_users[new_query] = datetime.now() + timedelta(seconds=20)
                    return render_template('login.html', msg="Invalid Credentials!!!")
    else:
        return render_template('login.html')


@app.route("/sqli_detection", methods=['GET', 'POST'])
def SQLi():
    if request.method == 'POST':
        
        new_query = request.form.get('sqlquery')
       
        if new_query is None or len(new_query) == 0:
            result = "Please provide a valid query."
        else:
            new_query_tokens = tokenizer([new_query])  
            new_query_feature = model.predict(new_query_tokens)  
            new_query_pred = svm_classifier.predict(new_query_feature)  
            if new_query_pred[0] == 1:
                result = "SQL Injection detected!"
                # count+=1
            else:
                result = "SQL Injection not detected."
                # count=0

        return render_template('index.html', query=new_query, res=result, acc= accuracy)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
