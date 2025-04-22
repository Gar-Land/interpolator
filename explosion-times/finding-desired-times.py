[]# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 13:59:57 2023

@author: Jerónimo García
"""
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta

arc_files = ["2016-231_12.29.58.500.ARC", "2016-232_11.29.59.350.ARC", 
             "2016-233_13.29.59.550.ARC", "2016-234_13.00.00.100.ARC", 
             "2016-235_11.59.59.500.ARC", "2016-236_12.59.59.450.ARC", 
             "2016-237_11.59.59.400.ARC", "2016-238_11.29.59.400.ARC", 
             "2016-239_11.29.58.650.ARC", "2016-240_12.59.59.100.ARC", 
             "2016-241_10.59.59.000.ARC", "2016-242_09.59.58.550.ARC", 
             "2016-242_14.09.58.450.ARC", "2016-243_07.54.57.750.ARC", 
             "2016-243_10.04.57.300.ARC", "2016-243_12.24.56.850.ARC", 
             "2016-244_08.49.58.950.ARC", "2016-244_11.44.58.700.ARC",
             "2016-244_15.44.58.750.ARC"]

explotion_dates = []

for file in arc_files:
    data = pd.read_csv(file,delim_whitespace=True,header=None)
    
    file_date = []
    explotion = np.argmax(data.iloc[:,3])
    date_of_explotion = data.iloc[explotion,0]
    year_of_explotion = date_of_explotion[0:4]
    day_number_of_explotion = date_of_explotion[5:8]
    hour_of_explotion = date_of_explotion[9:11]
    minute_of_explotion = date_of_explotion[13:14]
    
    day_number_of_explotion = day_number_of_explotion.rjust(3 + len(day_number_of_explotion), '0')
    template_date = datetime(int(year_of_explotion), 1, 1, int(hour_of_explotion), int(minute_of_explotion))
    resulting_object_date = template_date + timedelta(days=int(day_number_of_explotion) - 1)
    resulting_string_date = resulting_object_date.strftime("%Y-%#m-%#d-%#H-%#M")
    file_date.append(resulting_string_date)
    explotion_dates.append(tuple(file_date))