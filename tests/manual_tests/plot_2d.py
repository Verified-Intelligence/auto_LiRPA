import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.widgets import Slider
import numpy as np
import torch

from auto_LiRPA.bound_ops import BoundMul, BoundTanh, BoundSigmoid

function = 'tanh'
range_l = -3.
range_u = 3.
x_l = -1.
x_u = 1.


if function == 'x^2':
    def get_bound(x_l, x_u):
        a_l, _, b_l, a_u, _, b_u = BoundMul.get_bound_square(torch.tensor(x_l), torch.tensor(x_u))
        return a_l, b_l, a_u, b_u

    def f(x):
        return x * x
elif function == 'tanh' or function == 'sigmoid':
    if function == 'tanh':
        bound = BoundTanh('', '', '', {}, [], 0, 'cpu')
        def f(x):
            return np.tanh(x)
    elif function == 'sigmoid':
        bound = BoundSigmoid('', '', '', {}, [], 0, 'cpu')
        def f(x):
            return 1 / (1 + np.exp(-x))
    def get_bound(x_l, x_u):
        class Input(object):
            def __init__(self, x_l, x_u):
                self.lower = torch.tensor([[x_l]])
                self.upper = torch.tensor([[x_u]])
        # Create a fake input object with lower and upper bounds.
        i = Input(x_l, x_u)
        bound._init_linear(i)
        bound.bound_relax(i)
        return bound.lw.item(), bound.lb.item(), bound.uw.item(), bound.ub.item()

def fu(x, a, b):
    return a * x + b

def fl(x, a, b):
    return a * x + b

# Get initial values.
a_l, b_l, a_u, b_u = get_bound(x_l, x_u)
fig = plt.figure()
# Leave some space below for sliders.
plt.subplots_adjust(bottom=0.25)
ax = fig.gca()
x = np.linspace(range_l, range_u, 101)
# Plot main function.
plt.plot(x, f(x), color='skyblue', linewidth=1)
y_l, y_u = ax.get_ylim()
# Plot upper and lower bounds.
l_p, = plt.plot(x, fl(x, a_l, b_l), color='olive', linewidth=1, label="lb")
u_p, = plt.plot(x, fu(x, a_u, b_u), color='olive', linewidth=1, label="ub")
# Plot two straight lines for ub and lb.
l_pl, = plt.plot(x_l * np.ones_like(x), np.linspace(y_l, y_u, 101), color='blue', linestyle='dashed', linewidth=1)
u_pl, = plt.plot(x_u * np.ones_like(x), np.linspace(y_l, y_u, 101), color='blue', linestyle='dashed', linewidth=1)
plt.ylim(y_l, y_u)
plt.legend()

# Create sliders.
axcolor = 'lightgoldenrodyellow'
ax_xl = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
ax_xu = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)
s_xl = Slider(ax_xl, 'lb', range_l, range_u, valinit=x_l)
s_xu = Slider(ax_xu, 'ub', range_l, range_u, valinit=x_u)

def update_xu(val):
    # Update upper bound value, and update figure.
    global x_u
    x_u = val
    if x_u < x_l:
        print("x_u < x_l")
        return
    a_l, b_l, a_u, b_u = get_bound(x_l, x_u)
    u_p.set_ydata(fu(x, a_u, b_u))
    l_p.set_ydata(fl(x, a_l, b_l))
    u_pl.set_xdata(x_u * np.ones_like(x))
    fig.canvas.draw_idle()

def update_xl(val):
    # Update lower bound value, and update figure.
    global x_l
    x_l = val
    if x_u < x_l:
        print("x_u < x_l")
        return
    a_l, b_l, a_u, b_u = get_bound(x_l, x_u)
    u_p.set_ydata(fu(x, a_u, b_u))
    l_p.set_ydata(fl(x, a_l, b_l))
    l_pl.set_xdata(x_l * np.ones_like(x))
    fig.canvas.draw_idle()

s_xl.on_changed(update_xl)
s_xu.on_changed(update_xu)

plt.show()

7