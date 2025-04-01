import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


def get_data_loader(training=True):
    """
    TODO: implement this function.

    INPUT:
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if training:
        data = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        load = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
    else:
        data = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
        load = torch.utils.data.DataLoader(data, batch_size=64, shuffle=False)

    return load


def build_model():
    """
    TODO: implement this function.

    INPUT:
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model


def build_deeper_model():
    """
    TODO: implement this function.

    INPUT:
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10)
    )
    return model


def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT:
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy
        T - number of epochs for training

    RETURNS:
        None
    """
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(T):
        total, correct, run_loss = 0, 0, 0.0

        for data, target in train_loader:
            optimizer.zero_grad()
            outputs = model(data)

            loss = criterion(outputs, target)
            loss.backward()

            optimizer.step()
            run_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        loss_average = run_loss / total
        accuracy = (correct / total) * 100

        print(f"Train Epoch: {epoch} Accuracy: {correct}/{total}({accuracy:.2f}%) Loss: {loss_average:.3f}")


def evaluate_model(model, test_loader, criterion, show_loss=True):
    model.eval()

    accumulated_loss = 0.0
    correctly_classified = 0
    sample_count = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            output_logits = model(inputs)

            batch_loss = criterion(output_logits, labels)

            accumulated_loss += batch_loss.item() * inputs.shape[0]

            _, prediction = torch.max(output_logits, dim=1)

            sample_count += labels.shape[0]
            correctly_classified += (prediction == labels).sum().item()

    test_loss = accumulated_loss / sample_count
    test_accuracy = (correctly_classified / sample_count) * 100

    if show_loss:
        print(f"Average loss: {test_loss:.4f}")
    print(f"Accuracy: {test_accuracy:.2f}%")


def predict_label(model, test_images, index):
    fashion_categories = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

    target_image = test_images[index:index + 1]

    model.eval()
    with torch.no_grad():
        output = model(target_image)

        probability_distribution = F.softmax(output, dim=1)

        values, indices = torch.topk(probability_distribution, k=3, dim=1)

        top_values = values.flatten()
        top_indices = indices.flatten()

    for i in range(3):
        category = fashion_categories[top_indices[i]]
        confidence = top_values[i].item() * 100
        print(f"{category}: {confidence:.2f}%")


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    loss_fn = nn.CrossEntropyLoss()
    train_data = get_data_loader(training=True)
    test_data = get_data_loader(training=False)
    model_instance = build_model()
    train_model(model_instance, train_data, loss_fn, T=5)
    evaluate_model(model_instance, test_data, loss_fn, show_loss=True)
    sample_images, _ = next(iter(test_data))
    predict_label(model_instance, sample_images, index=0)