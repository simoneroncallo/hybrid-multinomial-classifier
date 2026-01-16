import torch

class SquareModulus(torch.nn.Module):
    """ Compute the square modulus of the input. """
    def forward(self, x):
        return torch.square(torch.abs(x))

class QuantumNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, use_bias_sigmoid, encoding = 'amplitude'):
        """ Initialize shallow network. 
        Arguments:
        input_dim -- Shape of the input array
        hidden_dim -- Number of hidden neurons
        """ 
        super().__init__()
        self.encoding = encoding
        self.bias_sigmoid = use_bias_sigmoid
        
        self.hidden_w = torch.nn.Parameter(torch.randn(hidden_dim, input_dim, dtype=torch.float64))
        self.output_w = torch.nn.Parameter(torch.randn(1, hidden_dim, dtype=torch.float64))
        
        self.activation = SquareModulus()
        if self.bias_sigmoid:
            self.bias = torch.nn.Parameter(torch.randn(1))
            self.sigmoid = torch.nn.Sigmoid()

        self.project() # Normalize

    def forward(self, x):
        """ Forward pass.
        Arguments:
        x -- Input (batch_size, num_features)
        W1 -> (num_features, num_hidden)
        W2 -> (1, num_hidden)
        """
        z = torch.matmul(x, self.hidden_w.t()) # Shape (batch_size, num_hidden)
        z = self.activation(z)
        z = torch.matmul(z, self.output_w.t()) # Shape (batch_size, 1)
        if self.bias_sigmoid:
            z += self.bias
            z = self.sigmoid(z)
        return z

    def project(self):
        """ Projection after optimizer.step()"""
        with torch.no_grad():
            norm = torch.sum(torch.square(self.hidden_w), dim=1).unsqueeze(1)
            self.hidden_w /= torch.sqrt(norm) # Normalize hidden layer (L2)
            self.output_w.copy_(torch.nn.functional.softplus(self.output_w)) # Positivity
            #norm = torch.sum(self.output_w)
            #self.output_w /= norm # Normalize output layer (L1)

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

        return (self.hidden_w, self.output_w), history_train, history_val
            
            

        