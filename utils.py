
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchsummary import summary
import torch.optim as optim
from torch_lr_finder import LRFinder
import torch.nn as nn

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def plot_image(train_loader):
    # get some random training images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)


    import torchvision
    # show images
    imshow(torchvision.utils.make_grid(images[:8]))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device

def model_summary(model):
    summary(model, input_size=(3, 32, 32))

def get_max_lr(model,train_loader,criterion):
    criterion = criterion
    optimizer = optim.Adam(model.parameters(), lr=1e-7, weight_decay=1e-2)
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    lr_finder.range_test(train_loader, end_lr=100, num_iter=100)
    xs, max_lr = lr_finder.plot() # to inspect the loss-learning rate graph
    lr_finder.reset() # to reset the model and optimizer to their initial state
    return max_lr, optimizer




def plot_prediction_image(test_loader,device,model):
    model.eval()
    with torch.no_grad():
        batch_data, batch_label = next(iter(test_loader))
        batch_data, batch_label = batch_data.to(device), batch_label.to(device)
        prediction = model(batch_data)
        prediction_class = np.argmax(prediction.cpu().numpy(), axis=1)
        classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        misclassified_images = []
        misclassified_labels = []
        misclassified_predictions = []
            # Check if the prediction is incorrect
        for i in range(len(prediction_class)): 
            if prediction_class[i] != batch_label[i].cpu().numpy():
            
                img = batch_data[i].clip(0, 1)#.astype(np.uint8)
                misclassified_images.append(img.cpu())
                misclassified_labels.append(classes[batch_label[i].item()])
                misclassified_predictions.append(classes[prediction_class[i]])

    fig = plt.figure(figsize=(14,7))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.tight_layout()
        img = misclassified_images[i].permute(1,2,0)
        class_name = misclassified_predictions[i]
        plt.imshow(img.cpu())
        plt.title(f'Actual: {misclassified_labels[i]} Predection: {class_name}')
        plt.xticks([])
        plt.yticks([])

    






