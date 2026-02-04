import math

class Value:
    def __init__(self, data, _child=(), _op = "", label = ""):
        self.data= data
        self.grad = 0
        self._backward = lambda : None # initially empty function.
        self._prev= set(_child)
        
        # below are for visualizing only.
        self._op = _op
        self.label = label

    # reprsentation
    def __repr__(self):
        return f"Value: {self.data:.3f}"

    # remember to output a 'Value' object only
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self,other), "+")

        # think about when '+' happens in compute graph
        # what is the _backward then?
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward # this is a pointer to the function. 
        # edit: it is not. this is weird.
        # we're trying to say : x = 32, y = x so y = 32.

        return out

    def __mul__ (self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self,other), "*")

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data 
        out._backward = _backward 

        return out

    def relu(self):

        self.data= max(self.data, 0)
        out = Value(self.data, (self, ), "relu")

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def tanh(self):

        x = math.exp(2 * self.data)
        x= (x - 1) / (x + 1)
        out = Value(x, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - x ** 2) * out.grad
        
        out._backward = _backward
        return out 

    def backward(self):
        vis = set()
        nodes = []
        def build(root):
            if root not in vis:
                vis.add(root)
                for child in root._prev:
                    build(child)
                nodes.append(root)

        self.grad = 1
        build(self)
        for v in reversed(nodes):
            v._backward()
    
    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return other + (-self)

    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,), f"**{other}")
        
        def _backward():
            self.grad = (other * self.data ** (other -1 )) * out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other