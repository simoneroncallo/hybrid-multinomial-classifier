import torch

class SquareModulus(torch.nn.Module):
    """ Compute the square modulus of the input. """
    def forward(self, x):
        return torch.square(torch.abs(x))

class QuantumNeuron(torch.nn.Module):
    def __init__(self, input_dim, use_bias_sigmoid, encoding = 'amplitude'):
        """ Initialize shallow network. 
        Arguments:
        input_dim -- Shape of the input array
        """ 
        super().__init__()
        self.encoding = encoding
        self.bias_sigmoid = use_bias_sigmoid
        
        self.weights = torch.nn.Parameter(torch.randn(input_dim, 1, dtype=torch.float64))
        
        self.activation = SquareModulus()
        if self.bias_sigmoid:
            self.bias = torch.nn.Parameter(torch.tensor([0.0]))
            self.sigmoid = torch.nn.Sigmoid()

        self.project() # Normalize

    def forward(self, x, eps = 1e-8):
        """ Forward pass.
        Arguments:
        x -- Input (Normalized)
        """
        if self.encoding == 'amplitude':
          weights = torch.sqrt(self.weights) # Amplitude
        if self.encoding == 'phase':
          weights = torch.exp(1j*self.weights)
        z = torch.matmul(x, self.weights)
        z = self.activation(z)
        if self.bias_sigmoid:
            z += self.bias
            z = self.sigmoid(z)
        return z

    def project(self):
        """ Projection after optimizer.step()"""
        with torch.no_grad():
          self.norm = torch.sum(torch.square(self.weights))
          self.weights /= torch.sqrt(self.norm)

    def train(self, train_loader, val_loader, num_epochs, user_loss, user_optimizer):
        """ Train the model"""
        history_train = torch.zeros((num_epochs,2), dtype=torch.float64)
        history_val = torch.zeros((num_epochs,2), dtype=torch.float64)
        for epoch in range(num_epochs):
            # Training
            num_items = 0
            for Xbatch, Ybatch in train_loader:
                user_optimizer.zero_grad() # Reset
                outputs = self.forward(Xbatch) # Predict
                loss = user_loss(outputs, Ybatch) # Loss
                loss.backward() # Gradients

                user_optimizer.step() # Update
                self.project()
        
                with torch.no_grad():
                    preds_train = (outputs >= 0.5).float() # Threshold
                    acc_train = (preds_train.eq(Ybatch).sum().item())
                    history_train[epoch,0] += loss.item() * Xbatch.size(0) # Loss
                    history_train[epoch,1] += acc_train # Accuracy
                    num_items += Xbatch.size(0)          
            history_train[epoch, :] /= num_items
                
            # Validation
            num_items = 0
            for Xbatch, Ybatch in val_loader:
                with torch.no_grad():
                    outputs_val = self.forward(Xbatch)
                    loss_val = user_loss(outputs_val, Ybatch)
                    preds_val = (outputs_val >= 0.5).float()
                    acc_val = (preds_val.eq(Ybatch).sum().item())
                    history_val[epoch, 0] += loss_val.item() * Xbatch.size(0) # Loss
                    history_val[epoch, 1] += acc_val # Accuracy
                    num_items += Xbatch.size(0)          
            history_val[epoch, :] /= num_items
    
        return self.weights, history_train, history_val
        