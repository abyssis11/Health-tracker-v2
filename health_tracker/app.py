from flask import Flask, render_template, redirect, url_for, request, jsonify, make_response
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, IntegerField, SelectField
from wtforms.validators import DataRequired, Length, Optional
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from kafka import KafkaConsumer
import pandas as pd
import json
import os
from flask_paginate import Pagination, get_page_parameter
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String, Float
import threading
import debugpy
from datetime import datetime
import atexit
import requests
from functools import wraps
import math
import re

app = Flask(__name__)
app.config['SECRET_KEY'] = 'health_tracker_app_key'
app.config['SESSION_COOKIE_NAME'] = 'health_tracker_session'
ML_MODEL_URI = os.getenv('ML_MODEL_URI')
API_TOKEN = os.getenv('API_TOKEN', 'my_secret_token')

# Database setup
DATABASE_URL = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

consumer_threads = {}

def convert_time_to_minutes(time_str):
    match = re.match(r'(\d+):(\d+):(\d+)', time_str)
    if match:
        hours, minutes, seconds = map(int, match.groups())
        return hours * 60 + minutes + seconds / 60
    else:
        raise ValueError(f"Invalid time format: {time_str}")

def send_data_to_ml_model(user_id, data):
    response = requests.post(f'{ML_MODEL_URI}/train', json={'user_id': user_id, 'data': data})
    if response.status_code == 200:
        print('Model trained successfully')
    else:
        print('Failed to train model')

def calculate_TRIMP(duration_minutes, average_hr, resting_hr, max_hr, gender):
    # Calculate HR reserve
    hr_reserve = max_hr - resting_hr
    # Calculate HR ratio
    hr_ratio = (average_hr - resting_hr) / hr_reserve
    # Constant b for male or female
    if gender == 'male':
        b = 1.92
    else:  # assuming gender == 'female'
        b = 1.67
    # Calculate TRIMPS
    TRIMPS = duration_minutes * hr_ratio * math.exp(b * hr_ratio)
    return TRIMPS

def require_api_token(func):
    @wraps(func)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('api-token')
        if token and token == API_TOKEN:
            return func(*args, **kwargs)
        else:
            return jsonify({'error': 'Unauthorized access'}), 403
    return decorated_function

class UserActivity(Base):
    __tablename__ = 'user_activities'
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, index=True)
    udaljenost = Column(Float)
    vrijeme = Column(Integer)  # Changed to Integer
    prosjecni_puls = Column(Float)
    ukupni_uspon = Column(Float)
    tezina = Column(Integer)

class User(Base, UserMixin):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String, nullable=False)
    max_heart_rate = Column(Integer, nullable=False, default=0)
    resting_heart_rate = Column(Integer, nullable=False, default=70)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def get_id(self):
        return str(self.username)
    
class UserAnalytics(Base):
    __tablename__ = 'user_analytics'
    username = Column(String, primary_key=True, index=True)
    avg_distance = Column(Float)
    avg_time = Column(Float)
    avg_heart_rate = Column(Float)
    avg_ascent = Column(Float)
    max_distance = Column(Float)
    max_time = Column(Float)
    max_ascent = Column(Float)

Base.metadata.create_all(bind=engine)

# Flask-Login setup
login_manager_health_tracker = LoginManager()
login_manager_health_tracker.init_app(app)
login_manager_health_tracker.login_view = 'login'

class HealthTrackerUser(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager_health_tracker.user_loader
def load_user(user_id):
    session = SessionLocal()
    user = session.query(User).filter_by(username=user_id).first()
    session.close()
    return user

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=25)])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=25)])
    password = PasswordField('Password', validators=[DataRequired()])
    age = IntegerField('Age', validators=[DataRequired()])
    gender = SelectField('Gender', choices=[('male', 'Male'), ('female', 'Female')], validators=[DataRequired()])
    max_heart_rate = IntegerField('Max Heart Rate', validators=[Optional()])
    resting_heart_rate = IntegerField('Resting Heart Rate', default=70, validators=[Optional()])
    submit = SubmitField('Register')

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        session = SessionLocal()
        user = User(username=form.username.data, age=form.age.data, gender=form.gender.data)
        user.set_password(form.password.data)
        
        if form.max_heart_rate.data:
            user.max_heart_rate = form.max_heart_rate.data
        else:
            user.max_heart_rate = 220 - form.age.data
        
        if form.resting_heart_rate.data:
            user.resting_heart_rate = form.resting_heart_rate.data
        
        session.add(user)
        session.commit()
        session.close()
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        session = SessionLocal()
        user = session.query(User).filter_by(username=form.username.data).first()
        session.close()
        if user and user.check_password(form.password.data):
            login_user(user = user, force = True)
            return redirect(url_for('index'))
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    user_id = current_user.username
    stop_kafka_consumer(user_id)
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    user_id = current_user.username
    if user_id not in consumer_threads or not consumer_threads[user_id][0].is_alive():
        start_kafka_consumer(user_id)
    
    session = SessionLocal()
    activities = session.query(UserActivity).filter(UserActivity.username == current_user.username).all()
    session.close()

    context = {"user": user_id, "page": 2, 'per_page': 4}

    return render_template('index.html', context=context)

@app.route('/activities', methods=['GET'])
@login_required
def get_activities():
    session = SessionLocal()

    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page',4, type=int)

    total_activities = session.query(UserActivity).filter(UserActivity.username == current_user.username).count()
    activities = session.query(UserActivity)\
        .filter(UserActivity.username == current_user.username)\
        .offset((page - 1) * per_page)\
        .limit(per_page)\
        .all()

    session.close()

    context = {
        'activities': activities,
        'total_activities': total_activities,
        'page': page,
        'per_page': per_page,
    }
    response = make_response(render_template('partials/activities.html', pagination=context))
    response.headers['new-endpoint'] = '/activities?page='+str(page)+'&per_page='+str(per_page)
    return response

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    user_id = current_user.username
    input_data = request.form.to_dict()  # Get form data as a dictionary
    input_data = {key: float(value) for key, value in input_data.items()}  # Convert values to float

    # Log input data for debugging
    print(f"Input data for prediction: {input_data}")

    response = requests.post(f'{ML_MODEL_URI}/predict', json={'user_id': user_id, 'input_data': input_data})
    if response.status_code == 200:
        prediction = response.json()['prediction']
        if prediction == 0:
            tezina = 'easy'
        elif prediction == 1:
            tezina = 'moderated'
        else:
            tezina = 'hard'
        return render_template('partials/prediction.html', prediction=tezina)
    else:
        app.logger.error(f"Prediction request failed: {response.text}")
        return jsonify({'error': 'Prediction failed'}), 500
    
@app.route('/user_activities/<user_id>', methods=['GET'])
@require_api_token
def get_user_activities(user_id):
    print(f"Request received for user activities: {user_id}")
    session = SessionLocal()
    activities = session.query(UserActivity).filter(UserActivity.username == user_id).all()
    session.close()
    
    activities_list = [{
        'Udaljenost': activity.udaljenost,
        'Vrijeme': activity.vrijeme,
        'Prosječni puls': activity.prosjecni_puls,
        'Ukupni uspon': activity.ukupni_uspon,
        'Tezina': activity.tezina
    } for activity in activities]
    
    print(f"Returning activities: {activities_list}")
    return jsonify({'activities': activities_list})

@app.route('/analytics')
@login_required
def get_analytics():
    user_id = current_user.username
    session = SessionLocal()
    analytics = session.query(UserAnalytics).filter(UserAnalytics.username == user_id).first()
    session.close()

    analytics_data = {}

    if analytics:
        if analytics.avg_distance is not None:
            analytics_data['avg_distance'] = round(analytics.avg_distance, 2)
        if analytics.avg_time is not None:
            analytics_data['avg_time'] = round(analytics.avg_time, 2)
        if analytics.avg_heart_rate is not None:
            analytics_data['avg_heart_rate'] = round(analytics.avg_heart_rate, 2)
        if analytics.avg_ascent is not None:
            analytics_data['avg_ascent'] = round(analytics.avg_ascent, 2)
        if analytics.max_distance is not None:
            analytics_data['max_distance'] = round(analytics.max_distance, 2)
        if analytics.max_time is not None:
            analytics_data['max_time'] = round(analytics.max_time, 2)
        if analytics.max_ascent is not None:
            analytics_data['max_ascent'] = round(analytics.max_ascent, 2)

    return render_template('partials/analytics.html', analytics=analytics_data)

def consume_kafka_messages(user_id, stop_event):
    consumer = KafkaConsumer(
        bootstrap_servers=os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        max_poll_records=10,
        auto_offset_reset='latest',
        enable_auto_commit=True,
        group_id=f'{user_id}-group',
        session_timeout_ms=6000,
        heartbeat_interval_ms=3000
    )
    topic = f'{user_id}-topic'
    consumer.subscribe([topic])

    session = SessionLocal()
    user = session.query(User).filter_by(username=user_id).first()
    try:
        while not stop_event.is_set():
            messages = consumer.poll(timeout_ms=1000)
            if not messages:
                print(f"No messages received for user {user_id}")
                continue

            for topic_partition, records in messages.items():
                new_data = []
                for message in records:
                    if stop_event.is_set():
                        break
                    data = message.value
                    print(f"Received message: {data}")

                    # Uncomment this line if filtering is required
                    if data['Vrsta aktivnosti'].lower().strip() not in ['trčanje', 'hodanje', 'planinarenje', 'terensko trčanje']:
                        continue

                    udaljenost = float(data['Udaljenost']) if data['Udaljenost'] != '--' else None
                    vrijeme = convert_time_to_minutes(data['Vrijeme']) if data['Vrijeme'] != '--' else None
                    prosjecni_puls = float(data['Prosječni puls']) if data['Prosječni puls'] != '--' else None
                    ukupni_uspon = None
                    if data['Ukupni uspon'] != '--':
                        value = data['Ukupni uspon']
                        if isinstance(value, str):
                            value = value.replace(',', '')
                        ukupni_uspon = float(value)
                    stres_score = float(data['Training Stress Score®']) if data['Training Stress Score®'] != 0.0 else None

                    if udaljenost == None or vrijeme == None or prosjecni_puls == None or ukupni_uspon == None:
                        tezina = None
                    else:
                        if stres_score != None:
                            if stres_score < 150:
                                tezina = 0
                            elif stres_score > 300:
                                tezina = 2
                            else:
                                tezina = 1
                        else:
                            TRIMPS = calculate_TRIMP(vrijeme, prosjecni_puls, user.resting_heart_rate, user.max_heart_rate, user.gender)
                            if TRIMPS < 70:
                                tezina = 0
                            elif TRIMPS > 140:
                                tezina = 2
                            else:
                                tezina = 1

                        new_data.append({
                            'Udaljenost': udaljenost,
                            'Vrijeme': vrijeme,
                            'Ukupni uspon': ukupni_uspon,
                            'Tezina': tezina
                        })

                    activity = UserActivity(
                        username=user_id,
                        udaljenost=udaljenost,
                        vrijeme=vrijeme,
                        prosjecni_puls=prosjecni_puls,
                        ukupni_uspon=ukupni_uspon,
                        tezina=tezina
                    )
                    session.add(activity)
                    session.commit()

                if new_data:
                    send_data_to_ml_model(user_id, new_data)
    except Exception as e:
        print(f"Error in consuming messages for user {user_id}: {e}")
    finally:
        consumer.close()
        session.close()


def start_kafka_consumer(user_id):
    stop_event = threading.Event()
    thread = threading.Thread(target=consume_kafka_messages, args=(user_id, stop_event), daemon=True)
    consumer_threads[user_id] = (thread, stop_event)
    thread.start()

def stop_kafka_consumer(user_id):
    if user_id in consumer_threads:
        _, stop_event = consumer_threads[user_id]
        stop_event.set()
        consumer_threads[user_id][0].join()
        del consumer_threads[user_id]

# Ensure all consumer threads are stopped when the application exits
@atexit.register
def shutdown():
    for user_id in list(consumer_threads.keys()):
        stop_kafka_consumer(user_id)

if __name__ == '__main__':
    #debugpy.listen(('0.0.0.0', 5678))  # Start debugpy on port 5678
    #print("Waiting for debugger attach...")
    #debugpy.wait_for_client()  # Pause execution until debugger is attached
    app.run(debug=True, host='0.0.0.0', port=5001)
