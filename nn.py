from value import Value
import random

lr = 0.05
epoch = 100

class neuron:
    def __init__(self, n_in, nonlin = True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(n_in)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __repr__(self):
        return f"{len(self.w)} : tanh"
    
    # how to call this class - forward pass in neuron
    def __call__(self, x):
        activation = sum((wi * xi for wi, xi in zip(self.w , x)),self.b)
        return activation.tanh() if self.nonlin else activation

    def parameters(self):
        return self.w + [self.b]

class Layer:

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP:

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"


def main():
    
    # 2 - 3 - 1

    n = MLP(2, [3,1])
    X = [[0,0], [0, 1], [1, 0], [1,1]]
    y = [0,1,1,0]

    print(n)

    ypred = [n(x) for x in X]
    print(f' these are initial loss: {ypred}')

    for i in range(epoch):
        # lets zero grad the parameters

        ypred = [n(x) for x in X]
        for p in n.parameters():
            p.grad = 0 

        loss = sum([(yp - yt)**2 for yt, yp in zip(y, ypred)])
        print(f'this is loss in epoch: {i}-> {loss}')
        loss.backward()


        for p in n.parameters():
            p.data += -lr * p.grad
    
    ypred = [n(x) for x in X]
    print(f"real values should be : [0,1,1,0]")
    print(f"training preds are: {ypred}")
















if __name__ == "__main__":
    main()