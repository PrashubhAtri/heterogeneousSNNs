import spikingjelly.activation_based as sj
import spikingjelly.activation_based.neuron as neuron
import spikingjelly.activation_based.functional as functional
import spikingjelly.activation_based.surrogate as surrogate

transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

def spike_latency_encoding(image, tau=50.0, threshold=0.2):
    image = image.squeeze(0)
    latencies = tau * torch.log(image / (image - threshold))
    latencies[image < threshold] = float('inf')
    return latencies

sample_img, _ = trainset[0]
spike_latencies = spike_latency_encoding(sample_img)

plt.imshow(spike_latencies.numpy(), cmap='hot')
plt.title("Spike Latency Encoding")
plt.colorbar()
plt.show()

for images, labels in trainloader:
    print(images.shape)
    break

for images, labels in testloader:
    print(images.shape)  
    break

class SNN(nn.Module):
    def __init__(self):
        super(SNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  
        self.lif1 = neuron.LIFNode(surrogate_function=surrogate.Sigmoid())
        self.fc2 = nn.Linear(128, 10)  
        self.lif2 = neuron.LIFNode(surrogate_function=surrogate.Sigmoid())

    def forward(self, x):
        x = x.view(x.shape[0], -1)  
        x = self.lif1(self.fc1(x))
        x = self.lif2(self.fc2(x))
        return x

model = SNN()
sample_input = torch.randn(64, 1, 28, 28)  
output = model(sample_input)
print(output.shape)  

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
time_steps = 10

for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True):
        optimizer.zero_grad()

        functional.reset_net(model)

        images = images.view(images.shape[0], -1)

        spike_inputs = torch.zeros((images.shape[0], time_steps, 784))
        latencies = spike_latency_encoding(images)

        for t in range(time_steps):
            spike_inputs[:, t, :] = (latencies.view(latencies.shape[0], -1) < (t + 1) * 5).float()

        outputs = model(spike_inputs.mean(dim=1))

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainset)}")


correct = 0
total = 0

with torch.no_grad():
    for images, labels in torch.utils.data.DataLoader(testset, batch_size=64):
        functional.reset_net(model)
        images = images.view(images.shape[0], -1)
        spike_inputs = torch.zeros((images.shape[0], time_steps, 28 * 28))
        latencies = spike_latency_encoding(images)

        for t in range(time_steps):
            spike_inputs[:, t, :] = (latencies.view(latencies.shape[0], -1) < (t + 1) * 5).float()

        outputs = model(spike_inputs.mean(dim=1))
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")
