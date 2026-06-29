import numpy as np
import matplotlib.pyplot as plt
import torch as torch
import torch.optim as optim
import softmax_analytical_edge as smae


def main():

    model = smae.SoftMaxAnalyticalEdge(
                                        temperature=1.0,
                                        noisy_start=False)

    # Create dummy dataset: y = x**2
    X = np.linspace(-1.0, 1.0, 500)

    # Y = np.sin(3*X)
    Y = np.log(X**4 + 1)
    # Y = np.sin(X*2)
    # Y = np.sin(2*(X-5))

    X_t = torch.from_numpy(X)
    Y_t = torch.from_numpy(Y)

    criterion = torch.nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    # optimizer = optim.LBFGS(model.parameters(), lr=0.0001, max_iter=100, history_size=150, line_search_fn='strong_wolfe')

    epochs = 50_000
    for epoch in range(1, epochs + 1):
        def closure():
            optimizer.zero_grad()
            y_pred = model(X_t)
            loss = criterion(y_pred, Y_t) / torch.mean(Y_t**2)  # relative MSE
            loss.backward()

            return loss

        loss = optimizer.step(closure)
        if epoch % 1000 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} loss={loss.item():.6e}")
        if loss.item() < 1e-3:
            print(f"Early stopping at epoch {epoch} with loss={loss.item():.6e}")
            break

    print("\nTrained parameters:")
    print("a_i:", model.a_i.detach().numpy())
    print("b_i:", model.b_i.detach().numpy())
    print("c_i:", model.c_i.detach().numpy())
    print("d_i:", model.d_i.detach().numpy())
    print("w_i:", model.w_i.detach().numpy())

    # compute final PI (softmax of w_i)
    PI = model.GetExpertProbabilities()
    print("Expert probability [X, X^2, x^3, tanh, sin] :", PI.detach().numpy())

    Y_pred = model(X_t).detach().numpy()

    # Print final loss
    final_loss = float(((Y_pred - Y) ** 2).mean())
    print(f"\nFinal MSE (numpy): {final_loss:.6e}")

    X_augmented = np.linspace(-1.5, 1.5, 500)
    Y_augmented = model(torch.from_numpy(X_augmented)).detach().numpy()

    # Plot and save comparison
    plt.figure(figsize=(8, 5))
    plt.plot(X, Y, label='reference')
    plt.plot(X, Y_pred, '--', label='model prediction')
    plt.plot(X_augmented, Y_augmented, ':', label='model prediction (augmented)')
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

