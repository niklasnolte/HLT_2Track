import re
from sys import argv

import numpy as np

with open(argv[1], 'r') as f:
    x = f.readlines()[-1]

x = " ".join(x.split(" ")[5:])
x = re.sub('x(\d)', 'x[:,\g<1>]', x)
x = x.replace("sin", "np.sin")
x = x.replace("cos", "np.cos")
x = x.replace("tan", "np.tan")
x = x.replace("asin", "np.asin")
x = x.replace("acos", "np.acos")
x = x.replace("atan", "np.atan")
x = x.replace("exp", "np.exp")

fun = eval("lambda x: " + x)

inp = np.random.rand(3, 4)

print(fun(inp))
