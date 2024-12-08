#RODRIGUEZ JAUREGUI JARED

import torch
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

# Cargar MNIST y dividir en lotes
train = datasets.MNIST('', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST('', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

# Visualizar un lote del conjunto de datos
for data in trainset:
    print(data)
    break

X, y = data[0][0], data[1][0]  # X: Imagen, y: Etiqueta.
print(data[1])

plt.imshow(data[0][0].view(28, 28))  # Visualizar la imagen.
plt.show()

# Verificar balance de clases
total = 0
counter_dict = {i: 0 for i in range(10)}  # Diccionario para contar cada dígito.

for data in trainset:
    Xs, ys = data
    for y in ys:
        counter_dict[int(y)] += 1
        total += 1

print(counter_dict)

# Mostrar porcentaje de cada clase
for i in counter_dict:
    print(f"{i}: {counter_dict[i] / total * 100.0}%")
