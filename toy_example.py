# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def RGB_to_hex(RGB):
    RGB = [int(x) for x in RGB]
    return "#" + "".join([
        "0{0:x}".format(v) if v < 16 else
        "{0:x}".format(v) for v in RGB])


def hex_to_RGB(hex):
    return [int(hex[i:i + 2], 16) for i in range(1, 6, 2)]


def linear_gradient(start_hex, finish_hex='#FFFFFF', n=10):
    s = hex_to_RGB(start_hex)
    f = hex_to_RGB(finish_hex)
    RGB_list = [RGB_to_hex(s)]
    for t in range(1, n):
        curr_vector = [
            int(s[j] + (float(t) / (n - 1)) * (f[j] - s[j])) for j in range(3)]
        RGB_list.append(RGB_to_hex(curr_vector))

    return RGB_list

update_rule = 'sgd_mom'
grid_length = 2
x_list = []
y_list = []
x = 0.5
y = 0.5
learning_rate = 1e-1
v_x = 0
v_y = 0
momentum = -0.9
for t in range(10000):
    z = x * y
    grad_x = y
    grad_y = -x
    if update_rule == 'sgd':
        x -= learning_rate * grad_x
        y -= learning_rate * grad_y
    elif update_rule == 'sgd_mom':
        grad_x = y
        v_x = learning_rate * grad_x + momentum * v_x
        x -= v_x
        grad_y = -x
        v_y = learning_rate * grad_y + momentum * v_y
        y -= v_y
    elif update_rule == 'consensus':
        grad_x += 0.5 * x
        grad_y += 0.5 * y
        x -= learning_rate * grad_x
        y -= learning_rate * grad_y
    elif update_rule == 'consensus_sgd_mom':
        v_x = learning_rate * grad_x + momentum * v_x
        v_y = learning_rate * grad_y + momentum * v_y
        x -= (v_x + learning_rate * 0.5 * x)
        y -= (v_y + learning_rate * 0.5 * y)
    x_list.append(x)
    y_list.append(y)

fig1 = plt.scatter(
    x_list, y_list, color=linear_gradient('#445ACC', '#b30000', t + 1), s=1)
my_mesh = np.array(
    np.meshgrid(
        np.linspace(-grid_length, grid_length, 20),
        np.linspace(-grid_length, grid_length, 20))).reshape((2, -1)).T
for i in range(len(my_mesh)):
    if 'consensus' in update_rule:
        grad_x = my_mesh[i][1] + 0.5 * my_mesh[i][0]
        grad_y = my_mesh[i][0] - 0.5 * my_mesh[i][1]
    else:
        grad_x = my_mesh[i][1]
        grad_y = my_mesh[i][0]
    plt.arrow(my_mesh[i][0], my_mesh[i][1],
              -grad_x / 15.0, grad_y / 15.0,
              head_width=0.02, head_length=0.02,
              color='gray')
# plt.grid()
plt.xlim([-grid_length, grid_length])
plt.ylim([-grid_length, grid_length])

if update_rule in ['sgd', 'consensus']:
    plt.savefig(update_rule + '.png', dpi=300)
else:
    plt.savefig('figs/' + update_rule + '_' + str(momentum) + '.png', dpi=300)
plt.show()
plt.draw()

