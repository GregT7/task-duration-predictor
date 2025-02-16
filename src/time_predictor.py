import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

productivity_event = pd.read_csv('../data/productivity_event.csv')
school_task = pd.read_csv('../data/school_task.csv')
assignment = pd.read_csv('../data/assignment.csv')
event_linking = pd.read_csv('../data/linking.csv')

def create_model(X_train, y_train, numeric_features, categorical_features):
    preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    # Define the model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor())
    ])

    # Train and evaluate the model
    model.fit(X_train, y_train)
    return model


p_event = pd.merge(productivity_event, event_linking, on='ID')
p_event = pd.merge(p_event, school_task, on='ID', how='left')


p_event['End'] = pd.to_datetime(p_event['End'], format="%I:%M %p")
p_event['Start'] = pd.to_datetime(p_event['Start'], format="%I:%M %p")

p_event['Duration'] = p_event['End'] - p_event['Start']
p_event['Duration'] = pd.to_timedelta(p_event['Duration'])
p_event['Duration'] = p_event['Duration'].dt.total_seconds() / 60



p_event = p_event[['ID', 'aID', 'Subject', 'Type', 'Duration']]

time_sum = p_event.groupby('aID').agg({'Duration': 'sum', 'ID': 'count'}).reset_index()

time_sum = time_sum.rename(columns={'Duration':'Sum', 'ID': 'Count'})


time_sum = pd.merge(time_sum, assignment, on='aID')
time_sum = time_sum[['aID', 'Class', 'Type', 'Name', 'fDifficulty', 'num_questions', 'Sum']]

class_filters = ['Crypto', 'OS', 'Graph', 'Teams', 'cTheory']
type_filters = ['Homework', 'Custom_Quiz', 'Quiz', 'Midterm']
time_sum = time_sum[time_sum['Class'].isin(class_filters)]
time_sum = time_sum[time_sum['Type'].isin(type_filters)]

s_event = pd.merge(school_task, event_linking, on="ID")
s_event = s_event[['aID', 'Comprehension', 'Focus', 'cDifficulty', 'Motivation']]
s_dict = {'Comprehension':'mean', 'Focus': 'mean', 'cDifficulty':'mean', 'Motivation': 'mean'}
assign_stats = s_event.groupby('aID').agg(s_dict).reset_index()
data = pd.merge(time_sum, assign_stats, on='aID')

features = ['Class', 'Type', 'fDifficulty', 'Sum']
target = 'Sum'
numeric_features = ['fDifficulty']
categorical_features = ['Class', 'Type']


print(data)
# X = data[features]
# y = data[target]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = create_model(X_train, y_train, numeric_features, categorical_features)

# y_pred = model.predict(X_test)
# mae = mean_absolute_error(y_test, y_pred)

# print(f'Mean Absolute Error: {mae:.2f} minutes')
# print(model.score(X_test, y_test))