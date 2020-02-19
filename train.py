from models.simpleConv import SimpleConv
import torch
import torch.nn

if __name__ == "__main__":
    net = SimpleConv()

    # Random input image
    input_mat = torch.randn(1,3,32,32)

    # Forward pass
    output = net(input_mat)    
    
    # Clear parameter buffer
    net.zero_grad()

    # Backpropagate
    output.backward(torch.randn(1,10))

    target = torch.randn(10)
    target = target.view(1, -1)
    criterion = nn.MSELoss()

    loss = criterion(output, target)
    print(loss)

