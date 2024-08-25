from matplotlib import pyplot as plt

with open("validation_finetuning_loss_log.txt", "r") as f:
    data = f.readlines()

x = []
y = []
for line in data:
    x.append(int(line.split()[3]) + int(line.split()[1]) * 1100)
    y.append(float(line.split()[-1]))

plt.plot(x, y)
plt.show()