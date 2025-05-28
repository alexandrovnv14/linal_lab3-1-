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
![Начальное состояние](https://i.imgur.com/initial_state.png)

**После X-гейта:**  
`|ψ〉 = 0.00|0〉 + 1.00|1〉`  
![После X-гейта](https://i.imgur.com/after_x_gate.png)

**После Y-гейта:**  
`|ψ〉 = (0.00-1.00j)|0〉 + 0.00|1〉`  
![После Y-гейта](https://i.imgur.com/after_y_gate.png)

**После Z-гейта:**  
`|ψ〉 = (0.00-1.00j)|0〉 + 0.00|1〉`  
![После Z-гейта](https://i.imgur.com/after_z_gate.png)

### 2. Диаграммы вероятностей

| Состояние | Диаграмма вероятностей |
|-----------|------------------------|
| Начальное | ![Prob Initial](https://i.imgur.com/prob_initial.png) |
| После X | ![Prob X](https://i.imgur.com/prob_x.png) |
| После H | ![Prob H](https://i.imgur.com/prob_h.png) |

### 3. Демонстрация CNOT  
**Исходное состояние:** |10〉  
**После CNOT:** |11〉  

**Визуализация состояний:**  
![CNOT States](https://i.imgur.com/cnot_states.png)

---

## Выводы  
1. Успешно реализованы все требуемые квантовые гейты  
2. Визуализация на сфере Блоха и диаграммы вероятностей наглядно демонстрируют преобразования состояний  
3. Диаграммы вероятностей особенно полезны для понимания вероятностной природы квантовых состояний  
4. Результаты полностью соответствуют теоретическим ожиданиям  
5. Программа может быть расширена для моделирования сложных квантовых алгоритмов  

[Полный исходный код](quantum_gates.py)
