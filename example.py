from MNISTDataset import get_mnist_dataset, SingleLabelMNIST, MultiLabelMNIST

whole_dataset = get_mnist_dataset(root="./data", train=True, download=True)

# Make dataset which has only data labelded as 0
mnist0 = SingleLabelMNIST(whole_dataset, 0)

# Make dataset which has data labelded as 0 and 1
mnist01 = MultiLabelMNIST(whole_dataset, [0, 1])
