
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import List


def generate_simple_plot(x: List, y: List, title: str="", x_label: str="",
        y_label: str="", y_lim: List[float]=[0.0, 1.0], save: bool=True,
        fname: str=""):
    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.set(xlabel=x_label, ylabel=y_label, ylim=y_lim, title=title)

    if save:
        fig.savefig("./plots/" + fname)
    else:
        plt.show()
