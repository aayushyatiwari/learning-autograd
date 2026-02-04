# This is minigrad

An autograd engine made from scratch inspired by Karpathy's micrograd       
This was a learning project for the internals of pytorch.       

there is an additional debug.py file for debuging.      

# Steps to use

- import `Value` object from value
- each value object will have some parameters that you can look at in value.py file
- main.py file was me playing with the Value class. You can use that file to generate graphs using `draw_dot()` method. It works in terminals too. Just install graphviz via pip and also `sudo apt install graphviz` for a system wide installation.
- for graphs, import them from graph.py file


### I would like everyone to feel what I felt when the neural network was working. It was a nice feeling. 


## TODO

- [x] switch ReLU → tanh

- [] port to C++
