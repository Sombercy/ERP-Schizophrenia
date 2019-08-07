import pandas
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import computations as comp

erps = pandas.read_csv('ERPdata.csv', index_col = 'subject')

demo = pandas.read_csv('demographic.csv', index_col = 'subject')

comp.pwr_time(erps, demo, 1, 150, 190, True)
#comp.rl_plt(erps, demo, 1, 1)
