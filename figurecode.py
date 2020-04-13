import pandas as pd
import glob
import re
import os

import matplotlib.pyplot as plt

filenames = glob.glob('*')
for name in filenames:
    if re.search('\.',name):
        filenames.remove(name)
    else:
        None


all_data_names = {}
for name in filenames:
    for root, dirs, files in os.walk(name):
        all_data_names[name] = files

jh_df_names = []
jh_df = []
for name in all_data_names['johns_hopkins_csse']:
    jh_df_names.append(re.split('19-', name)[-1])
    jh_df.append(pd.read_csv('johns_hopkins_csse/' + name))

jh_df_names = [name[:-4] for name in jh_df_names]
jh_data = dict(zip(jh_df_names,jh_df))
print(jh_df_names)





state_cases = jh_data['daily-dashboard-cases-by-states'][['province_state','country_region','confirmed','deaths','active']]

state_cases = state_cases[state_cases['country_region'] == 'US']
state_cases.sort_values(by=['confirmed'],inplace=True, ascending=False)



plt.figure(figsize=(12,12))
plt.bar(state_cases['province_state'],state_cases['confirmed'],width=0.8, color = 'blue', alpha = 0.8, label = 'Confirmed Cases')
plt.bar(state_cases['province_state'],state_cases['deaths'],width=0.8, color = 'red', label = 'Deaths')
plt.xticks(rotation=90)
plt.annotate('120000', xytext = (2,18000), xy = (0,18500), arrowprops=dict(color='orange', linewidth=1, headwidth=10, headlength=8))
plt.annotate('35000', xytext = (3,15000), xy = (1,15500), arrowprops=dict(color='orange', linewidth=1, headwidth=10, headlength=8))
plt.ylim(0,20000)
plt.xlabel('US States')
plt.ylabel('Cases of COVID-19')
plt.title('COVID-19 Cases in the US', fontdict=dict(fontsize=13))
plt.legend(loc = 'upper right', fontsize = 'large')
plt.savefig('covid_us.jpg')

