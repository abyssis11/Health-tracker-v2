from flask import Flask, render_template, redirect, url_for, request
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Length
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from kafka import KafkaProducer
import pandas as pd
import json
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'activity_app_key'
app.config['SESSION_COOKIE_NAME'] = 'activity_session'  # Different session cookie name

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=25)])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User(form.username.data)
        login_user(user)
        return redirect(url_for('index'))
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join('data', current_user.id, file.filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            file.save(filepath)
            send_to_kafka(current_user.id, filepath)
            return 'File uploaded and processed'
    return render_template('index.html')

def send_to_kafka(user_id, filepath):
    producer = KafkaProducer(
        bootstrap_servers=os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode('utf-8')
    )
    topic = f'{user_id}-topic'
    data = pd.read_csv(filepath, encoding='utf-8')
    for _, row in data.iterrows():
        message = row.to_dict()
        producer.send(topic, value=message)
    producer.flush()


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)