import pandas as pd
import glob
import re
import os

from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
from bokeh.io import output_file, show
from bokeh.layouts import column, row
from bokeh.io import curdoc
from bokeh.models import LinearColorMapper
import numpy as np

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
country_cases = jh_data['daily-dashboard-cases-by-country'][['country_region','confirmed','deaths','recovered','active']]
country_cases.sort_values(by=['confirmed'],inplace=True, ascending=True)


listofsizes = np.array(country_cases['confirmed'].tolist())

source = ColumnDataSource(data={
    'x'       : country_cases['confirmed'],
    'y'       : country_cases['deaths'],
    'country' : country_cases['country_region'],
    'sizes'    : np.sqrt(listofsizes)/2
})

color_mapper = LinearColorMapper(palette='Turbo256', low=min(country_cases['deaths']), high=max(country_cases['deaths']))


p = figure(plot_width=600, plot_height=600, title = 'Cases of COVID-19 in the World', x_axis_label = 'confirmed cases', y_axis_label = 'deaths', tools = 'box_select,pan,wheel_zoom,box_zoom,reset', x_range=[min(country_cases['confirmed']),max(country_cases['confirmed'])+35000] , y_range =[min(country_cases['deaths']),max(country_cases['deaths'])+2000])

hover = HoverTool(tooltips=[('Country', '@country'),('Number of Confirmed Cases', '@x'),('Number of Deaths', '@y')])

p.add_tools(hover)


p.circle(x='x', y='y', size = 'sizes', source=source, color={'field': 'y', 'transform': color_mapper})

##################################

listofsizes = np.array(country_cases['confirmed'].tolist())

source = ColumnDataSource(data={
    'x'       : country_cases['confirmed'],
    'y'       : country_cases['recovered'],
    'country' : country_cases['country_region'],
    'sizes'    : np.sqrt(listofsizes)/2
})

color_mapper = LinearColorMapper(palette='Viridis256', low=min(country_cases['recovered']), high=max(country_cases['recovered']))


p1 = figure(plot_width=600, plot_height=600, title = 'Cases of COVID-19 in the World', x_axis_label = 'confirmed cases', y_axis_label = 'recoveries', tools = 'box_select,pan,wheel_zoom,box_zoom,reset', x_range=[min(country_cases['confirmed']),max(country_cases['confirmed'])+35000] , y_range =[min(country_cases['recovered']),max(country_cases['recovered'])+2000])

hover = HoverTool(tooltips=[('Country', '@country'),('Number of Confirmed Cases', '@x'),('Number of Recoveries', '@y')])

p1.add_tools(hover)


p1.circle(x='x', y='y', size = 'sizes', source=source, color={'field': 'y', 'transform': color_mapper})


layout = row(p, p1)

output_file('cor.html')
show(layout)






