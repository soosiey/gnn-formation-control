import matplotlib.pyplot  as plt

model=[]
loss=[]
with open("../document.csv") as file:
    while True:
        line=file.readline()
        if not line:
            break
        model.append(int(line.strip().split(",")[0]))
        loss.append(float(line.strip().split(",")[1]))
plt.figure(figsize=(7, 4))
plt.title("Loss")
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.plot(model, loss)
plt.savefig("loss_fig.png")