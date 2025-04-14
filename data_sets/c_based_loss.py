import torch
import matplotlib.pyplot as plt
#from torch.autograd.functional import hessian
from torch.func import  vmap


def train(model, dataloader, optimizer, epochs):
    Ctarget = torch.tensor([[ 7.16042643e+06,  4.27997603e+02, -2.18278728e-11],
                     [ 4.27997603e+02,  7.14548324e+06,  7.09405867e-11],
                     [-2.18278728e-11,  7.09405867e-11,  3.57285779e+06]])
    model.train()

    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    losses = []
    plot_epochs = []

    def psi_fn(E_voigt):
        E_voigt_reshaped = E_voigt.view(1,1,3)
        psi = model.EvaluatePsi(E_voigt_reshaped)
        return psi.squeeze()

    def compute_hessian_manual(psi_fn, x):
        """
        psi_fn: function that takes input of shape (1, 3) and returns scalar
        strain: tensor of shape (B, 1, 3)

        Returns:
            H: tensor of shape (B, 3, 3), each is the Hessian of psi wrt the 3D input
        """
        B = strain.shape[0]
        H_all = []

        for b in range(B):
            x_b = strain[b]  # shape (1, 3)
            x_b = (x_b.detach().squeeze()).requires_grad_(True)

            H_b = torch.autograd.functional.hessian(psi_fn, x_b, create_graph=True)  # shape (1, 3)

            H_all.append(H_b)
        return torch.stack(H_all)  # shape (B, 3, 3)

    initial_loss = None
    for epoch in range(epochs):
        epoch_loss = 0.0
        for strain_history in dataloader:
            strain = strain_history[0].detach().requires_grad_(True)

            #Ccalculated = compute_hessian_vectorized(model,strain)
            Ccalculated = compute_hessian_manual(psi_fn, strain)

            loss = torch.norm(Ctarget[:]-Ccalculated)**2

            if initial_loss==None:
                initial_loss=loss.item()


            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Store loss
            losses.append(loss.item()/initial_loss)
            plot_epochs.append(epoch)

        # Update plot
        if epoch%1 == 0:
            ax.clear()
            ax.plot(plot_epochs, losses, label="Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            plt.pause(0.01)  # Pause to update the plot

            print(Ccalculated)

            #print("W",model.convex_nn.Wdiag)
            print("W",model.convex_nn.W)
            print("a",model.convex_nn.a_b)
            print("d",model.convex_nn.d_e)

        # print("epoch_loss",epoch_loss)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()/(initial_loss)}")
        # if epoch%10==0:
        #     print(0.5*(model.symmetric_layer.C + model.symmetric_layer.C.T))

        # if epoch%20 == 0:
        #     print("raw",model.convex_nn.raw_weight1,model.convex_nn.raw_weight2,model.convex_nn.raw_weight3)

    ax.clear()
    ax.plot(plot_epochs, losses, label="Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.pause(0.01)  # Pause to update the plot

    fig = ax.get_figure()  # Get the parent Figure object
    fig.savefig("c_based_loss_plot.png")
    plt.close(fig)  # If you want to free memory

    plt.ioff()  # Turn off interactive mode
    #plt.show()