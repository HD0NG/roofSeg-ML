# %%
import torch
import time
import torch.nn as nn
import torch.optim as optim

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA GPU is available.")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Apple MPS is available.")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU.")

# %%
# Define a more complex model
class ComplexModel(nn.Module):
    def __init__(self):
        super(ComplexModel, self).__init__()
        self.layer1 = nn.Linear(2000, 4000)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Linear(4000, 2000)
        self.layer4 = nn.ReLU()
        self.layer5 = nn.Linear(2000, 1000)
        self.layer6 = nn.ReLU()
        self.layer7 = nn.Linear(1000, 500)
        self.layer8 = nn.ReLU()
        self.layer9 = nn.Linear(500, 100)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        return x

# Initialize model, loss function, and optimizer
model = ComplexModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Generate larger random data
input_data = torch.randn(10000, 2000).to(device)
target = torch.randn(10000, 100).to(device)

# Warm-up
print("Warming up...")
for _ in range(20):
    output = model(input_data)
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Benchmark
num_iterations = 100
print(f"Running benchmark for {num_iterations} iterations...")
start_time = time.time()

for i in range(num_iterations):
    output = model(input_data)
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (i + 1) % 10 == 0:
        print(f"Completed {i + 1} iterations")

end_time = time.time()

# Calculate and print results
total_time = end_time - start_time
avg_time_per_iteration = total_time / num_iterations
print(f"\nDevice: {device}")
print(f"Total time: {total_time:.4f} seconds")
print(f"Average time per iteration: {avg_time_per_iteration:.4f} seconds")
print(f"Iterations per second: {1 / avg_time_per_iteration:.2f}")



