import pandas as pd

p_event = pd.read_csv('../data/pevent.csv')
new_p_event = pd.read_csv('../data/new_pevent.csv')
assignment = pd.read_csv('../data/assignment.csv')
new_assignment = pd.read_csv('../data/new_assignment.csv')
event_linking = pd.read_csv('../data/linking.csv')
new_event_linking = pd.read_csv('../data/new_linking.csv')

school_task = pd.read_csv('../data/school_task.csv')


p_event = pd.concat([p_event, new_p_event], ignore_index=True, sort=False)
assignment = pd.concat([assignment, new_assignment], ignore_index=True, sort=False)
event_linking = pd.concat([event_linking, new_event_linking], ignore_index=True, sort=False)

