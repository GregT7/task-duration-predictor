import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import datetime as datetime

def extract_weighted_energy(e_df, main_df):
    energy = e_df[['aID', 'Energy']]
    l_energy = energy[energy['Energy'] == 'l']
    m_energy = energy[energy['Energy'] == 'm']
    h_energy = energy[energy['Energy'] == 'h']
    low_count = l_energy.groupby('aID').agg({'Energy': 'count'}).reset_index()
    medium_count = m_energy.groupby('aID').agg({'Energy': 'count'}).reset_index()
    high_count = h_energy.groupby('aID').agg({'Energy': 'count'}).reset_index()
    main_df = pd.merge(main_df, low_count, on="aID", how="left").rename(columns={'Energy': 'l'})
    main_df = pd.merge(main_df, medium_count, on="aID", how="left").rename(columns={'Energy': 'm'})
    main_df = pd.merge(main_df, high_count, on="aID", how="left").rename(columns={'Energy': 'h'})
    main_df = main_df.fillna(0)
    main_df['m'] = main_df['m'] * 5
    main_df['h'] = main_df['h'] * 10
    return main_df[['h', 'm', 'l']].sum(axis=1)

productivity_event = pd.read_csv('../data/pevent.csv')
new_productivity_event = pd.read_csv('../data/new_pevent.csv')
assignment = pd.read_csv('../data/assignment.csv')
new_assignment = pd.read_csv('../data/new_assignment.csv')
event_linking = pd.read_csv('../data/linking.csv')
new_event_linking = pd.read_csv('../data/new_linking.csv')
school_task = pd.read_csv('../data/school_task.csv')

productivity_event = pd.concat([productivity_event, new_productivity_event], ignore_index=True, sort=False)
assignment = pd.concat([assignment, new_assignment], ignore_index=True, sort=False)
event_linking = pd.concat([event_linking, new_event_linking], ignore_index=True, sort=False)

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
p_event['End'] = pd.to_datetime(p_event['End'], format="%I:%M %p")
p_event['Start'] = pd.to_datetime(p_event['Start'], format="%I:%M %p")
p_event['Duration'] = p_event['End'] - p_event['Start']
p_event['Duration'] = pd.to_timedelta(p_event['Duration'])
p_event['Duration'] = p_event['Duration'].dt.total_seconds() / 60
p_event = p_event[['ID', 'aID', 'Subject', 'Type', 'Duration']]


time_sum = p_event.groupby('aID').agg({'Duration': 'sum'}).reset_index()
time_sum = time_sum.rename(columns={'Duration':'Sum'})
time_sum = pd.merge(time_sum, assignment, on='aID')
class_filters = ['Digital Design', 'OP Systems', 'CORG', 'Data SA', 'Databases',
                   'Circuits', 'DBMS', 'Comparative', 'Teams I', 'Embedded', 'SWE']
class_filters += ['OS', 'Graph', 'cTheory', 'Crypto']
type_filters = ['Quiz', 'Exam', 'Lab', 'Project', 'Homework']
time_sum = time_sum[time_sum['Class'].isin(class_filters)]
data = time_sum[time_sum['Type'].isin(type_filters)]
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

features = ['Class', 'Type']
target = 'Sum'
numeric_features = []
categorical_features = ['Class', 'Type']

X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = create_model(X_train, y_train, numeric_features, categorical_features)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(f'Mean Absolute Error: {mae:.2f} minutes')
print(model.score(X_test, y_test))

new_data = pd.DataFrame({
    'Class': ['DBMS'],  # Example class
    'Type': ['Quiz']
})

predicted_time = model.predict(new_data)
print(f'Predicted time: {predicted_time[0]:.2f} minutes')