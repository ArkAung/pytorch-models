from models.simpleConv import SimpleConv
import torch
import torch.nn as nn
import torch.optim as optim

if __name__ == "__main__":
    net = SimpleConv()

    # Random input image
    input_mat = torch.randn(1,3,32,32)

    # Forward pass
    output = net(input_mat)    
    
    # Clear gradients otherwise the gradients will be accumulated
    # net.zero_grad()

    # Backpropagate with respect to output - this is actually just for the sake of demo 
    # You have you backgpropagate with respect to loss
    # output.backward(torch.randn(1,10))

    # Random target
    target = torch.randn(10)

    # Modify the dimension of target so that it matches with neural network output
    target = target.view(1, -1)

    # Define the type of loss function
    criterion = nn.MSELoss()

    # Calculate the loss between output and target
    loss = criterion(output, target)

    # Zero the gradient buffers of all parameters
    net.zero_grad()

    print('conv1.bias.grad before backprop')
    print(net.conv1.bias.grad)

    loss.backward()

    print('conv1.bias.grad after backprop')
    print(net.conv1.bias.grad)

    # create your optimizer - Stochastic Gradient Descent
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    optimizer.zero_grad() # zero the gradient buffers -- because of the above line this is same as saying net.zero_grad()
                          # another difference is that if you have different optimizers for different parts of the model, 
                          # model.zero_grad() would clear all parameters of the model, while the optimizer.zero_grad() call will just 
                          # clean the gradients of the parameters that were passed to it.
    output = net(input_mat)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step() # Does gradient update, which updates the weights

