import torch
import matplotlib.pyplot as plt

def train(model, dataloader, optimizer, epochs):

    model.train()

    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    losses = []
    plot_epochs = []

    initial_loss = None
    for epoch in range(epochs):
        epoch_loss = 0.0
        for strain_history, stress_history, target_work in dataloader:
            # strain_history = strain_history.to(device)
            # stress_history = stress_history.to(device)
            #target_work = target_work.to(device)

            strain_rate = strain_history.detach().clone()
            strain_rate[:,1:,:] = strain_history[:,1:,:] - strain_history[:,0:-1,:]
            # strain_rate = strain_rate.to(device)

            strain_history = strain_history.requires_grad_(True)
            # strain_history = strain_history.to(device)


            # Forward pass
            predicted_stress_history = model(strain_history)
            err_stress = stress_history - predicted_stress_history
            err_work_aux = torch.sum(err_stress*strain_rate,axis=2)
            err_work = err_work_aux[:,-1] #to minimize the last one
            #err_work = torch.cumsum(err_work_aux, dim=1) #ensure that we accumulate the error over time
            loss = torch.norm(err_work)**2

            #loss = torch.norm(stress_history - predicted_stress_history)
            if initial_loss==None:
                initial_loss=loss.item()
            # print("initial_loss",initial_loss)
            # print("loss",loss)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            # scheduler.step()
            #print("raw",model.convex_nn.raw_weight1,model.convex_nn.raw_weight2,model.convex_nn.raw_weight3)

            epoch_loss += loss.item()

            # Store loss
            losses.append(loss.item()/initial_loss)
            plot_epochs.append(epoch)

        # Update plot
        if epoch%20 == 0:
            ax.clear()
            ax.plot(plot_epochs, losses, label="Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            plt.pause(0.1)  # Pause to update the plot

            #print("W",model.convex_nn.Wdiag)
            print("W",model.convex_nn.W)
            print("a",model.convex_nn.a_b)
            print("d",model.convex_nn.d_e)

        # print("epoch_loss",epoch_loss)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()/(initial_loss)}")

    #finalize
    ax.clear()
    ax.plot(plot_epochs, losses, label="Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.pause(0.01)  # Pause to update the plot

    fig = ax.get_figure()  # Get the parent Figure object
    fig.savefig("work_based_loss_plot.png")
    plt.close(fig)  # If you want to free memory

    plt.ioff()  # Turn off interactive mode

    #plt.show()