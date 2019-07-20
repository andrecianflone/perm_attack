
import os
import torch
import numpy
import torch.nn as nn
import sinkhorn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import model

dir_path = os.path.dirname(os.path.realpath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#parameters
n_numbers = 50
lr = 0.1
temperature = 1.0
batch_size = 10
# samples_per_num = 5
samples_per_num = 1
n_iter_sinkhorn = 10
n_units =32
noise_factor= 1.0
dropout_prob = 0.
num_iters = 500
n_epochs = 301

# Training process
def train_model(model, criterion, optimizer, batch_size, n_numbers, n_epochs):
    # Get data
    train_ordered, train_random, train_hard_perms =\
                    sinkhorn.sample_uniform_and_order(batch_size, n_numbers)

    x = train_random.to(device).detach()
    perms = train_hard_perms.to(device).detach()
    y = train_ordered.to(device).detach()

    loss_history = []
    epoch_history = []

    for epoch in range(n_epochs):
        epoch_history.append(epoch)
        # Training phase
        model.train()

        optimizer.zero_grad()

        # Predict permutation and correct ordered input
        soft_perms, x_ordered = model(x)
        loss= criterion(y, x_ordered)

        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        # Update the progress bar.
        if epoch % 50 == 0:
            print("Epoch {0:03d}: l2 loss={1:.4f}".format(epoch,
                                                            loss_history[-1]))
    #save the model for evaluation
    torch.save(model.state_dict(), os.path.join(dir_path, 'trained_model'))
    print('Training completed')
    return loss_history, epoch_history

# Create the neural network
model = model.Sinkhorn_Net(\
        latent_dim= n_units,
        output_dim= n_numbers,
        temp=temperature,
        noise_factor = noise_factor,
        n_iter_sinkhorn = n_iter_sinkhorn,
        dropout_prob = dropout_prob).to(device)

n_params = 0
for p in model.parameters():
    n_params += numpy.prod(p.size())
print('# of parameters: {}'.format(n_params))

# Loss
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8)

# Train
loss_history, epoch_history = train_model(model, criterion, optimizer, batch_size, n_numbers, n_epochs=n_epochs)

# Plot
plt.plot(epoch_history, loss_history)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Training Loss')
plt.grid(True)
plt.savefig("training_loss.png")
plt.show()

