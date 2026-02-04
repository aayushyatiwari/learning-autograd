from value import Value
from graph import  draw_dot

def main():
    # weights and biases - one neuron
    x1 = Value(2, label='x1')
    w1 = Value(-3, label='w1')
    x2 = Value(0,label='x2')
    w2 = Value(1,label = 'w2')
    b = Value(6.8771); b.label = 'b' 

    x1w1 = x1 * w1; x1w1.label = 'x1*w1'
    x2w2 = x2 * w2; x2w2.label = 'x2*w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label='x1w1+x2w2'
    z = x1w1x2w2 + b; z.label = 'z'
    a = z.tanh()
    a.grad = 1.0; a.label = 'a'


    '''
    if you want to see backprop in action, 
    comment the below line and then check all the 
    .grad attributes. 

    and then uncomment it and then see all the 
    .grad attributes.
    
    '''

    a.backward() # the line that does backprop

    '''
    below is the code for manual backprop 
    '''
    # print("Forward pass")
    # print(a)
    # print(z)
    # print(x1w1,x2w2,b)
    # print(x1,w1)
    # print(x2,w2)
    # # backward passes
    # print("Backprop")
    # a._backward()
    # z._backward()
    # x1w1._backward()
    # x2w2._backward()
    # print(a)
    # print(z)
    # print(x1w1,x2w2,b)
    # print(x1,w1)
    # print(x2,w2)

    # sudo apt install graphviz
    # pip install graphviz
    d = draw_dot(a)
    d.render("graph", view=True) 
if __name__ == "__main__":
    main()
