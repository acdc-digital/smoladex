from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///medex.db'
db = SQLAlchemy(app)

@app.route('/')
def home():
    return 'Hello, Medex!'

if __name__ == '__main__':
    app.run(debug=True)