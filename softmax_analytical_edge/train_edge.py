import numpy as np
import matplotlib.pyplot as plt
import torch as torch
import torch.optim as optim
import softmax_analytical_edge as smae


def main():

    model = smae.SoftMaxAnalyticalEdge()

    # Create dummy dataset: y = x**2
    X = np.linspace(0, 1.0, 100, dtype=np.float32)

    Y = ((X) ** 1.5).astype(np.float32)
    # Y = np.log(X**2 + 1)
    # Y = np.sin(X*2)

    X_t = torch.from_numpy(X)
    Y_t = torch.from_numpy(Y)

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 100_000
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        y_pred = model(X_t)
        loss = criterion(y_pred, Y_t)
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} loss={loss.item():.6e}")

    print("\nTrained parameters:")
    print("a_i:", model.a_i.detach().numpy())
    print("b_i:", model.b_i.detach().numpy())
    print("c_i:", model.c_i.detach().numpy())
    print("d_i:", model.d_i.detach().numpy())
    # print("w_i:", model.w_i.detach().numpy())

    # compute final PI (softmax of w_i)
    with torch.no_grad():
        exp_w = torch.sum(torch.exp(model.w_i))
        PI = torch.exp(model.w_i) / exp_w
        print("Expert probability:", PI.detach().numpy())

        Y_pred = model(X_t).numpy()

    # Print final loss
    final_loss = float(((Y_pred - Y) ** 2).mean())
    print(f"\nFinal MSE (numpy): {final_loss:.6e}")

    # Plot and save comparison
    plt.figure(figsize=(8, 5))
    plt.plot(X, Y, label='reference')
    plt.plot(X, Y_pred, '--', label='model prediction')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('SoftMaxAnalyticalEdge fit')
    out_file = 'train_result.png'
    plt.savefig(out_file)
    print(f"Saved comparison plot to {out_file}")
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()

