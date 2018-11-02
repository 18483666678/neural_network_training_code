# Readme
Add date: 20181101

## Reference
https://pytorch.org/  
https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html

## File list
training_a_classifier.py : Demo base on CIFAR10 dataset.

## Requirements
python 3  
pytorch  
numpy  
matplotlib  

## Recommendation
Strong recommend you use Pycharm + Anaconda to develop.

## Key points
1. training_a_classifier.py
```python
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root="./data", train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
...
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```