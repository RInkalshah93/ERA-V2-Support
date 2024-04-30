
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchsummary import summary
import torch.optim as optim
from torch_lr_finder import LRFinder
import torch.nn as nn
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

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

def get_max_lr(model,train_loader,criterion,device):
    criterion = criterion
    optimizer = optim.Adam(model.parameters(), lr=1e-7, weight_decay=1e-2)
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
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

    fig = plt.figure(figsize=(20,8))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        #plt.tight_layout()
        img = misclassified_images[i].permute(1,2,0)
        class_name = misclassified_predictions[i]
        plt.imshow(img.cpu())
        plt.title(f'Actual: {misclassified_labels[i]} Predection: {class_name}')
        plt.xticks([])
        plt.yticks([])
    return misclassified_images,misclassified_labels,misclassified_predictions,prediction

def plot_grad_cam(misclassified_images,misclassified_labels,misclassified_predictions,prediction,model):
    target_layers = [model.layer3[-1]]
    input_tensor = torch.stack(misclassified_images)

    # Create an input tensor image for your model.
    # Note: input_tensor can be a batch tensor with several images!
    #print(input_tensor.shape)
    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers)

    # You can also use it within a with statement, to make sure it is freed,
    # In case you need to re-create it inside an outer loop:
    # with GradCAM(model=model, target_layers=target_layers) as cam:
    #   ...

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category
    # will be used for every image in the batch.
    # Here we use ClassifierOutputTarget, but you can define your own custom targets
    # That are, for example, combinations of categories, or specific outputs in a non standard model.
    misclassified_target = []
    for prediction in misclassified_predictions:
        targets = ClassifierOutputTarget(classes.index(prediction))
        misclassified_target.append(targets)
        
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=misclassified_target)
    #print(grayscale_cam.shape)

    # In this example grayscale_cam has only one image in the batch:
    # grayscale_cam = grayscale_cam[0, :]
    # print(grayscale_cam.shape)
    # visualization = show_cam_on_image(np.array(misclassified_images), grayscale_cam, use_rgb=True)

    # You can also get the model outputs without having to re-inference
    model_outputs = cam.outputs


    fig = plt.figure(figsize=(20,8))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        #plt.tight_layout()
        img = np.transpose(misclassified_images[i], (1,2,0))
        class_name = misclassified_predictions[i]
        grayscale_i = grayscale_cam[i, :]
        visualization = show_cam_on_image(np.array(img), grayscale_i, use_rgb=True)
        plt.imshow(visualization)
        plt.title(f'Actual: {misclassified_labels[i]} Predection: {class_name}')
        plt.xticks([])
        plt.yticks([])

        






