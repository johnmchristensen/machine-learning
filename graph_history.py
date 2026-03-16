import matplotlib.pyplot as plt

def graph_history(history, title = ""):
    epochs = range(1, len(history.train_metrics) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history.train_metrics, label='Training Accuracy')
    plt.plot(epochs, history.valid_metrics, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(title if title != '' else 'Training and Validation Accuracy per Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
