from flask import Flask
from flask_bcrypt import Bcrypt
from flask_login import LoginManager
from flask_sslify import SSLify
from config import Config

app = Flask(__name__)
app.config.from_object(Config)
sslify = SSLify(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'localhost.torus.login'
