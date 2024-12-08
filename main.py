#RODRIGUEZ JAUREGUI JARED 

import torch  # Biblioteca principal para trabajar con tensores y redes neuronales.
from torchvision import transforms, datasets  # Manejo de conjuntos de datos y transformaciones.
import matplotlib.pyplot as plt  # Herramienta para graficar y visualizar datos.
import torch.nn as nn  # Permite definir modelos y capas de redes neuronales.
import torch.nn.functional as F  # Proporciona funciones como activaciones y pérdida.
import torch.optim as optim  # Algoritmos de optimización para entrenar redes.

# **Cargando los datos MNIST**
# `train`: Datos de entrenamiento. `test`: Datos de prueba.
train = datasets.MNIST('', train=True, download=True, 
                        transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST('', train=False, download=True, 
                        transform=transforms.Compose([transforms.ToTensor()]))

# **Preparar los datos en lotes pequeños para entrenar más fácilmente**
# `batch_size=10` significa que se manejarán 10 imágenes a la vez.
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)  # Mezclar datos en cada iteración.
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)  # Mantener el orden para pruebas.

# **Definición de la red neuronal**
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Capas totalmente conectadas:
        # 1. Toma imágenes de tamaño 28x28 (784 píxeles) y las reduce a 64 neuronas.
        self.layer1 = nn.Linear(28*28, 64)
        # 2. Conecta 64 neuronas a otras 64.
        self.layer2 = nn.Linear(64, 64)
        # 3. Otra conexión entre 64 neuronas.
        self.layer3 = nn.Linear(64, 64)
        # 4. Finalmente, reduce a 10 salidas (una para cada dígito: 0-9).
        self.layer4 = nn.Linear(64, 10)

    def forward(self, x):
        # **Pasar los datos a través de las capas y activaciones:**
        # 1. Aplicar función ReLU para agregar no linealidad.
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        # 2. No usar ReLU en la última capa, porque queremos probabilidades.
        x = self.layer4(x)
        # 3. Convertir las salidas en probabilidades logarítmicas con softmax.
        return F.log_softmax(x, dim=1)

# Crear un modelo de red neuronal.
net = Net()
print(net)  # Mostrar la estructura de la red.

# **Configurar el entrenamiento**
# Usar la función de pérdida CrossEntropy, que compara predicciones con las etiquetas correctas.
loss_function = nn.CrossEntropyLoss()

# Configurar el optimizador Adam para ajustar los pesos de la red.
optimizer = optim.Adam(net.parameters(), lr=0.001)

# **Entrenamiento de la red**
for epoch in range(2):  # Realizar 2 iteraciones completas sobre los datos de entrenamiento.
    for data in trainset:  # Tomar lotes de datos.
        X, y = data  # `X`: imágenes, `y`: etiquetas (números reales).
        net.zero_grad()  # Reiniciar los gradientes antes de calcular.
        output = net(X.view(-1, 784))  # Aplanar las imágenes de 28x28 a 784.
        loss = F.nll_loss(output, y)  # Calcular la pérdida entre predicción y etiqueta real.
        loss.backward()  # Propagar la pérdida hacia atrás.
        optimizer.step()  # Ajustar los pesos según los gradientes.
    print(loss)  # Imprimir la pérdida al final de cada época.

# **Evaluación del modelo**
correct = 0  # Contador de predicciones correctas.
total = 0  # Contador total de muestras evaluadas.
with torch.no_grad():  # No calcular gradientes para ahorrar memoria.
    for data in testset:
        X, y = data
        output = net(X.view(-1, 784))  # Aplanar las imágenes para pasarlas por la red.
        for idx, i in enumerate(output):
            # `torch.argmax(i)`: Encuentra la clase más probable.
            if torch.argmax(i) == y[idx]:  # Comparar predicción con etiqueta real.
                correct += 1
            total += 1

# Mostrar la precisión del modelo.
print("Exactitud: ", round(correct / total, 3))

# **Visualización de una predicción**
plt.imshow(X[0].view(28, 28))  # Mostrar una imagen del conjunto de prueba.
plt.show()
print(torch.argmax(net(X[0].view(-1, 784))[0]))  # Mostrar la predicción de la red para esa imagen.
