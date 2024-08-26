from matplotlib import pyplot as plt

with open("validation_finetuning_loss_log.txt", "r") as f:
    data = f.readlines()

x = []
y = []
for line in data:
    x.append(int(line.split()[3]) + int(line.split()[1]) * 1100)
    y.append(float(line.split()[-1]))

with open("validation_loss_log.txt", "r") as f:
    data = f.readlines()

x1 = []
y1 = []
for line in data:
    x1.append(int(line.split()[0]))
    y1.append(float(line.split()[-1]))

plt.title("Pretraining Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.grid()
plt.plot(x1, y1)
plt.savefig("pretraining_loss")