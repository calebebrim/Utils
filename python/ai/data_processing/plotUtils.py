from bokeh.plotting import figure, output_file, show
import numpy as np
import numpy.matlib 
from bokeh.palettes import Spectral11
from bokeh.models import HoverTool

def plot(y,x=None, plot_type='-', plot_width=1000, plot_height=500):
    print(y,x)
    plt = figure(plot_width=plot_width, plot_height=plot_height)
    x = list(range(len(y))) if x == None else x
    if plot_type == '-':
        plt.line(y=y, x=x)
    show(plt)





def multi_plot(y, x=None,legends=None, plot_type='-', plot_width=1000, plot_height=500,launch=True):
    if(type(y) == list):
        y = np.array(y)
    hover = HoverTool(tooltips=[
        ("dx", "@x"),
        ("dy", "@y")
    ])
    plt = figure(plot_width=plot_width, plot_height=plot_height,
                 tools=[hover])
    x = np.matlib.repmat(
        (list(range(y.shape[1]))), m=y.shape[0], n=1) if x == None else x
    legends = [""]*len(y) if legends == None else legends
    # print(legends)
    # print(y, y.shape)
    # print(x, x.shape)
    mypalette = Spectral11[:]

    if plot_type == '-':
        for i in range(len(y)):
            data = dict(x=x[i],y=y[i])
            plt.line(y='y', x='x',legend=legends[i],line_color=mypalette[i%len(mypalette)],source=data)
        
    if launch:
        show(plt)
    return plt

