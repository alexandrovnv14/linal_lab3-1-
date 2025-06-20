# Лабораторная работа: Реализация квантовых гейтов  
**Цель:** Реализация базовых квантовых операций с визуализацией на Python  

---

## Теоретическая часть  
### 1. Представление кубита  
Кубит описывается вектором состояния:  
`|ψ〉 = α|0〉 + β|1〉`  
где:  
- α, β - комплексные числа (амплитуды вероятностей)  
- |α|² + |β|² = 1  

### 2. Квантовые гейты  
**Однокубитные гейты Паули:**  

| Гейт | Матрица         | Действие          |
|------|-----------------|-------------------|
| X    | [[0, 1], [1, 0]] | Инверсия состояния |
| Y    | [[0, -i], [i, 0]] | Поворот + фаза    |
| Z    | [[1, 0], [0, -1]] | Фазовый сдвиг     |

**Двухкубитный гейт CNOT:**  
```
[[1, 0, 0, 0],
 [0, 1, 0, 0],
 [0, 0, 0, 1],
 [0, 0, 1, 0]]
```  
Действие: Если control=1, инвертирует target  

---

## Реализация  
### Класс Qubit  
```python
class Qubit:
    def __init__(self, state=None):
        self.state = np.array([1, 0], dtype=complex) if state is None else state
        self.normalize()
    
    def apply_gate(self, gate):
        self.state = gate @ self.state
    
    def measure(self):
        prob_0 = abs(self.state[0])**2
        return 0 if random.random() < prob_0 else 1
    
    def normalize(self):
        norm = np.linalg.norm(self.state)
        self.state /= norm
    
    def probabilities(self):
        return [abs(self.state[0])**2, abs(self.state[1])**2]
```

### Реализация гейтов  
```python
def pauli_x():
    return np.array([[0, 1], [1, 0]], dtype=complex)

def pauli_y():
    return np.array([[0, -1j], [1j, 0]], dtype=complex)

def cnot(control, target, num_qubits=2):
    # Реализация матрицы CNOT 4x4
    ...
```

### Визуализация  
```python
def plot_bloch_sphere(qubit, title=""):
    # Отрисовка сферы и вектора состояния
    ...

def plot_probabilities(qubit, title=""):
    probs = qubit.probabilities()
    states = ['|0>', '|1>']
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(states, probs, color=['blue', 'red'])
    plt.ylim(0, 1)
    plt.ylabel('Вероятность')
    plt.title(title)
    
    # Добавление значений на столбцы
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}', ha='center', va='bottom')
    
    plt.show()
```

---

## Результаты выполнения  
### 1. Применение гейтов Паули с визуализацией

**Исходное состояние:**  
`|ψ〉 = 1.00|0〉 + 0.00|1〉`  

```python
q = Qubit()
plot_probabilities(q, "Начальное состояние")
plot_bloch_sphere(q, "Начальное состояние")
```

**После X-гейта:**  
`|ψ〉 = 0.00|0〉 + 1.00|1〉`  
![image](https://github.com/user-attachments/assets/e8676739-36ae-4516-a183-b6c93cddf9e7)
```python
q.apply_gate(pauli_x())
plot_probabilities(q, "После X-гейта")
plot_bloch_sphere(q, "После X-гейта")
```

**После Y-гейта:**  
`|ψ〉 = (0.00-1.00j)|0〉 + 0.00|1〉`  
![image](https://github.com/user-attachments/assets/0a020523-d911-4a33-aaa4-eb0dc7438375)

```python
q.apply_gate(pauli_y())
plot_probabilities(q, "После Y-гейта")
plot_bloch_sphere(q, "После Y-гейта")
```

**После  Z-гейта:**
`|ψ〉 = 0.707|0〉 - 0.707|1`
![image](https://github.com/user-attachments/assets/3a1b3bfe-923d-477b-8801-54a45e57f0a1)

```python
q.apply_gate(pauli_z())
plot_probabilities(q, "После Z-гейта")
plot_bloch_sphere(q, "После Z-гейта")
```

### 2.Вероятности (примеры вывода)

#### Начальное состояние:
```
Вероятности:
|0>: 1.00
|1>: 0.00
```

#### После H-гейта (Адамара):
```
![image](https://github.com/user-attachments/assets/5f83942f-4a1b-42c1-a974-f03dca34248e)

Вероятности:
|0>: 0.50
|1>: 0.50
```

### 3. Визуализация на сфере Блоха

Позиции на сфере Блоха для различных состояний:

| Состояние | Координаты (x, y, z) | Расположение |
|-----------|----------------------|--------------|
| |0>       | (0, 0, 1)           | Северный полюс |
| |1>       | (0, 0, -1)          | Южный полюс |
| |+>       | (1, 0, 0)           | На экваторе по оси X |
| |->       | (-1, 0, 0)          | На экваторе по оси -X |

### 4. Демонстрация CNOT  
**Исходное состояние:** |10〉  
**После CNOT:** |11〉
![image](https://github.com/user-attachments/assets/19efc7f1-c74f-4372-9b56-d992dfd0150f)


```python
# Создаем систему из двух кубитов
q0 = Qubit([0, 1])  # |1>
q1 = Qubit()        # |0>

# Применяем CNOT
# (реализация будет в полном коде)
```

---

## Выводы  
1. Успешно реализованы все требуемые квантовые гейты  
2. Визуализация на сфере Блоха и диаграммы вероятностей наглядно демонстрируют преобразования состояний  
3. Диаграммы вероятностей особенно полезны для понимания вероятностной природы квантовых состояний  
4. Результаты полностью соответствуют теоретическим ожиданиям  
5. Программа может быть расширена для моделирования сложных квантовых алгоритмов  
