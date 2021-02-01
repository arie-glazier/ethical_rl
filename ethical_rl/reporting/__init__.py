import sys
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from ethical_rl.constants import *

class Reporter:
  def __init__(self, **kwargs):
    self.results_destination = kwargs[RESULTS_DESTINATION]

    self.plt = plt

    self.plot_colors = ['r','g','b','c','m','y','k','w']

  def clear_plot(self):
    self.plt.clf()

  def plot(self, y, x=None, color=None, label=None):
    if x == None: x = range(0,len(y))
    args = [x,y]
    if color: args.append(color)
    kwargs = {}
    if label: kwargs["label"] = label
    self.plt.plot(*args, **kwargs)

  def make_graph(self, data, x_label, y_label, title):
    self.clear_plot()
    self.plt.xlabel(x_label)
    self.plt.ylabel(y_label)
    self.plt.title(title)
    self.plot(data)

  def show_graph(self):
    self.plt.show()

  def save_figure(self, filename):
    self.plt.savefig(f"{self.results_destination}/{filename}.png")

  def make_stacked_graph(self, data_list, x_label, y_label, title, legend_location):
    self.clear_plot()
    color_generator = (x for x in self.plot_colors)
    self.plt.xlabel(x_label)
    self.plt.ylabel(y_label)
    self.plt.title(title)
    for label, data in data_list: 
      self.plot(y=data, color=next(color_generator), label=label)
    self.plt.legend(loc=legend_location)