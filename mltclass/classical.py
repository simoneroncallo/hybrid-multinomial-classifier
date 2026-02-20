import torch

class ClassicalNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, use_bias_sigmoid, num_layers: int = 1, encoding = 'amplitude', device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32):
        """ 
        Initialize a classical neural network. For simplicty, all the hidden layers have the same number of neurons.
        Arguments:
        input_dim -- Shape of the input array
        hidden_dim -- Number of hidden neurons
        """ 
        super().__init__()
        self.encoding = encoding
        self.bias_sigmoid = use_bias_sigmoid
        self.layers = []
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        
        # Hidden layers
        dim = input_dim
        for _ in range(num_layers):
            self.layers.append(torch.nn.Dropout(p=0.1))
            self.layers.append(torch.nn.Linear(dim, hidden_dim, bias = use_bias_sigmoid, device = self.device, dtype = self.dtype))
            self.layers.append(torch.nn.BatchNorm1d(hidden_dim, device = self.device, dtype = self.dtype)) # Batch normalization
            self.layers.append(torch.nn.ReLU())
            dim = hidden_dim
        # Output layer
        self.layers.append(torch.nn.Linear(dim, 1, bias = use_bias_sigmoid, device = self.device, dtype = self.dtype))
        self.layers.append(torch.nn.Sigmoid())
        
        self.model = torch.nn.Sequential(*self.layers) # Unroll the list of layers as arguments of nn.Sequential
        

    def forward(self, x):
        """ 
        Forward pass using the Sequential model.
        """
        z = self.model(x)
        return z

    def fit(self, train_loader, val_loader, num_epochs, user_loss, user_optimizer, user_scheduler):
        """ Train the model"""
        history_train = torch.zeros((num_epochs,2), device = "cpu", dtype = torch.float32)
        history_val = torch.zeros((num_epochs,2), device = "cpu", dtype = torch.float32)
        for epoch in range(num_epochs):
            # Training
            self.train()
            num_items = 0
            for Xbatch, Ybatch in train_loader:
                # Ensure device and dtype are correct
                Xbatch = Xbatch.to(device = self.device, dtype = self.dtype, non_blocking = True)
                Ybatch = Ybatch.to(device = self.device, dtype = self.dtype, non_blocking = True)
                
                user_optimizer.zero_grad() # Reset
                outputs = self.forward(Xbatch) # Predict
                loss = user_loss(outputs, Ybatch) # Loss
                loss.backward() # Gradients
                user_optimizer.step() # Update
        
                with torch.no_grad():
                    preds_train = (outputs >= 0.5).to(dtype = Ybatch.dtype) # Threshold
                    acc_train = (preds_train.eq(Ybatch).sum().item())
                    history_train[epoch,0] += loss.item() * Xbatch.size(0) # Loss
                    history_train[epoch,1] += acc_train # Accuracy
                    num_items += Xbatch.size(0)          
            
            user_scheduler.step()
            history_train[epoch, :] /= num_items
                
            # Validation
            self.eval()
            num_items = 0
            with torch.inference_mode():
                for Xbatch, Ybatch in val_loader:
                    # Ensure device and dtype are correct
                    Xbatch = Xbatch.to(device = self.device, dtype = self.dtype, non_blocking = True)
                    Ybatch = Ybatch.to(device = self.device, dtype = self.dtype, non_blocking = True)
                    
                    outputs_val = self.forward(Xbatch)
                    loss_val = user_loss(outputs_val, Ybatch)
                    preds_val = (outputs_val >= 0.5).to(dtype = Ybatch.dtype)
                    acc_val = (preds_val.eq(Ybatch).sum().item())
                    history_val[epoch, 0] += loss_val.item() * Xbatch.size(0) # Loss
                    history_val[epoch, 1] += acc_val # Accuracy
                    num_items += Xbatch.size(0)          
                    
            history_val[epoch, :] /= num_items

        return (None, None), history_train, history_val
            
            

        
