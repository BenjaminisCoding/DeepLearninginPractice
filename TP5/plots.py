import matplotlib.pyplot as plt

def plot_results(train_step, train_ep, test_ep):
    
    # Plotting the training step losses
    plt.figure(figsize=(12, 6))
    
    # First subplot for training step losses
    plt.subplot(1, 2, 1)
    plt.plot(train_step['total_loss'], label='Total Loss per Step')
    plt.plot(train_step['loss1'], label='Loss 1 per Step')
    plt.plot(train_step['loss2'], label='Loss 2 per Step')
    plt.plot(train_step['loss3'], label='Loss 3 per Step')
    plt.title('Training Step Losses')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    
    # Second subplot for training and testing epoch losses
    plt.subplot(1, 2, 2)
    plt.plot(train_ep['total_loss'], label='Total Training Loss per Epoch')
    plt.plot(test_ep['total_loss'], label='Total Testing Loss per Epoch', linestyle='--')
    plt.title('Training and Testing Epoch Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_results_2(train_step, train_ep, test_ep):
    
    plt.figure(figsize=(20, 12))

    # First row: Each loss for train_ep and test_ep
    losses = ['total_loss', 'loss1', 'loss2', 'loss3']
    for i, loss_type in enumerate(losses, 1):
        plt.subplot(2, 4, i)
        plt.plot(train_ep[loss_type], label=f'Train {loss_type}')
        plt.plot(test_ep[loss_type], label=f'Test {loss_type}', linestyle='--')
        plt.title(f'{loss_type.capitalize()} per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

    # Second row: Each loss from train_step
    for i, loss_type in enumerate(losses, 5):
        plt.subplot(2, 4, i)
        plt.plot(train_step[loss_type], label=f'{loss_type} per Step')
        plt.title(f'{loss_type.capitalize()} per Step')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()

