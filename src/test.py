import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import datetime as datetime

# def extract_weighted_energy(e_df, main_df):
#     energy = e_df[['aID', 'Energy']]
#     l_energy = energy[energy['Energy'] == 'l']
#     m_energy = energy[energy['Energy'] == 'm']
#     h_energy = energy[energy['Energy'] == 'h']
#     low_count = l_energy.groupby('aID').agg({'Energy': 'count'}).reset_index()
#     medium_count = m_energy.groupby('aID').agg({'Energy': 'count'}).reset_index()
#     high_count = h_energy.groupby('aID').agg({'Energy': 'count'}).reset_index()
#     main_df = pd.merge(main_df, low_count, on="aID", how="left").rename(columns={'Energy': 'l'})
#     main_df = pd.merge(main_df, medium_count, on="aID", how="left").rename(columns={'Energy': 'm'})
#     main_df = pd.merge(main_df, high_count, on="aID", how="left").rename(columns={'Energy': 'h'})
#     main_df = main_df.fillna(0)
#     main_df['m'] = main_df['m'] * 5
#     main_df['h'] = main_df['h'] * 10
#     return main_df[['h', 'm', 'l']].sum(axis=1)

productivity_event = pd.read_csv('../data/new_pevent.csv')
school_task = pd.read_csv('../data/school_task.csv')
assignment = pd.read_csv('../data/new_assignment.csv')
event_linking = pd.read_csv('../data/new_linking.csv')

p_event = pd.merge(productivity_event, event_linking, on='ID')
p_event = pd.merge(p_event, school_task, on='ID', how='left')

p_event['End'] = pd.to_datetime(p_event['End'], format="%I:%M %p")
p_event['Start'] = pd.to_datetime(p_event['Start'], format="%I:%M %p")

p_event['Duration'] = p_event['End'] - p_event['Start']
p_event['Duration'] = pd.to_timedelta(p_event['Duration'])
p_event['Duration'] = p_event['Duration'].dt.total_seconds() / 60


# p_event = p_event[['ID', 'aID', 'Subject', 'Type', 'Energy', 'Duration']]
time_sum = p_event.groupby('aID').agg({'Duration': 'sum', 'ID': 'count'}).reset_index()
time_sum = time_sum.rename(columns={'Duration':'Sum', 'ID': 'Count'})
time_sum = pd.merge(time_sum, assignment, on='aID')
# time_sum = time_sum[['aID', 'Class', 'Type', 'Name', 'fDifficulty', 'num_questions', 'Due', 'Sum']]


# time_sum['w_sum'] = extract_weighted_energy(p_event, time_sum)

# energy counting - idea 2


class_filters = ['Crypto', 'OS', 'Graph', 'Teams', 'cTheory']
type_filters = ['Homework', 'Custom_Quiz', 'Quiz', 'Exam']
time_sum = time_sum[time_sum['Class'].isin(class_filters)]
data = time_sum[time_sum['Type'].isin(type_filters)]

# time_sum['Due'] = pd.to_datetime(time_sum['Due'], format="%m/%d/%Y")
# start_date = datetime.datetime(2025, 1, 13)
# time_sum['Days'] = (time_sum['Due'] - start_date).dt.days


# s_event = pd.merge(school_task, event_linking, on="ID")
# s_event = s_event[['aID', 'Comprehension', 'Focus', 'cDifficulty', 'Motivation']]
# s_dict = {'Comprehension':'mean', 'Focus': 'mean', 'cDifficulty':'mean', 'Motivation': 'mean'}
# assign_stats = s_event.groupby('aID').agg(s_dict).reset_index()

# data = pd.merge(time_sum, assign_stats, on='aID')
# data.drop(columns=['aID', 'Due'], inplace=True)
# cols = {'fDifficulty': 'fDiff', 'num_questions': 'n_Qs', 'Comprehension': 'Compr', 'cDifficulty': 'cDiff', 'Motivation': 'Motiv'}
# data.rename(columns=cols, inplace=True)

# data.Compr = data.Compr.round()
# data.Focus = data.Focus.round()
# data.cDiff = data.cDiff.round()
# data.Motiv = data.Motiv.round()
# print(data)

print(data.corr(numeric_only=True))
plt.figure(figsize=(14, 12))
sns.heatmap(data.corr(numeric_only=True), annot=True)
plt.show()