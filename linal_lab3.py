import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi
from typing import List, Tuple

class Qubit:
    def __init__(self, state: np.ndarray = None):
        """
        Инициализация кубита. По умолчанию в состоянии |0⟩
        :param state: Вектор состояния в виде numpy массива [a, b], где |a|^2 + |b|^2 = 1
        """
        if state is None:
            self.state = np.array([1, 0], dtype=complex)
        else:
            if len(state) != 2:
                raise ValueError("State vector must have length 2")
            norm = np.linalg.norm(state)
            if norm != 0:
                self.state = state / norm
            else:
                self.state = np.array([1, 0], dtype=complex)
    
    def apply_gate(self, gate: np.ndarray):
        """
        Применение однокубитного гейта к кубиту
        :param gate: Унитарная матрица 2x2
        """
        self.state = np.dot(gate, self.state)
    
    def measure(self) -> int:
        """
        Измерение кубита (коллапс в |0⟩ или |1⟩)
        :return: 0 или 1
        """
        prob_0 = np.abs(self.state[0]) ** 2
        outcome = np.random.choice([0, 1], p=[prob_0, 1 - prob_0])
        self.state = np.array([1, 0] if outcome == 0 else [0, 1], dtype=complex)
        return outcome
    
    def get_bloch_coordinates(self) -> Tuple[float, float, float]:
        """
        Получение координат кубита на сфере Блоха
        :return: (x, y, z) координаты
        """
        a, b = self.state
        # Вычисление сферических координат
        theta = 2 * np.arccos(np.abs(a))
        phi = np.angle(b) - np.angle(a)
        
        # Преобразование в декартовы координаты
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return x, y, z
    
    def __str__(self) -> str:
        a, b = self.state
        # Красивое форматирование комплексных чисел
        def format_complex(c):
            real = f"{c.real:.3f}".rstrip('0').rstrip('.') if c.real != 0 else ""
            imag = f"{c.imag:.3f}".rstrip('0').rstrip('.') if c.imag != 0 else ""
            
            if real and imag:
                return f"{real}+{imag}i"
            elif real:
                return real
            elif imag:
                return f"{imag}i"
            else:
                return "0"
        
        a_str = format_complex(a)
        b_str = format_complex(b)
        return f"{a_str}|0⟩ + {b_str}|1⟩"

class QuantumSystem:
    @staticmethod
    def pauli_x() -> np.ndarray:
        """Гейт Паули X (аналог классического NOT)"""
        return np.array([[0, 1], [1, 0]], dtype=complex)
    
    @staticmethod
    def pauli_y() -> np.ndarray:
        """Гейт Паули Y"""
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    
    @staticmethod
    def pauli_z() -> np.ndarray:
        """Гейт Паули Z"""
        return np.array([[1, 0], [0, -1]], dtype=complex)
    
    @staticmethod
    def hadamard() -> np.ndarray:
        """Гейт Адамара"""
        return np.array([[1, 1], [1, -1]], dtype=complex) / sqrt(2)
    
    @staticmethod
    def cnot(control: int, target: int, num_qubits: int = 2) -> np.ndarray:
        """
        Гейт CNOT (управляемый NOT)
        :param control: Индекс управляющего кубита (0 или 1)
        :param target: Индекс целевого кубита (0 или 1)
        :param num_qubits: Общее количество кубитов (по умолчанию 2)
        :return: Матрица гейта CNOT
        """
        if num_qubits < 2:
            raise ValueError("CNOT gate requires at least 2 qubits")
        
        if control == target:
            raise ValueError("Control and target qubits must be different")
        
        # Базисные состояния для N кубитов
        basis_states = 2 ** num_qubits
        cnot_matrix = np.eye(basis_states, dtype=complex)
        
        # Меняем местами |10⟩ и |11⟩ (если control=0, target=1)
        for i in range(basis_states):
            # Получаем бинарное представление состояния
            binary = format(i, f'0{num_qubits}b')
            if binary[control] == '1':
                # Если управляющий кубит равен 1, инвертируем целевой кубит
                target_bit = binary[target]
                new_target_bit = '1' if target_bit == '0' else '0'
                new_binary = binary[:target] + new_target_bit + binary[target+1:]
                j = int(new_binary, 2)
                # Меняем местами строки в матрице
                cnot_matrix[i, i] = 0
                cnot_matrix[j, i] = 1
        
        return cnot_matrix

def plot_bloch_sphere(qubits: List[Qubit], title: str = ""):
    """Визуализация кубитов на сфере Блоха"""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Рисуем сферу
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='b', alpha=0.1)
    
    # Оси и подписи
    ax.quiver([0], [0], [0], [1.5], [0], [0], color='k', arrow_length_ratio=0.1)
    ax.quiver([0], [0], [0], [0], [1.5], [0], color='k', arrow_length_ratio=0.1)
    ax.quiver([0], [0], [0], [0], [0], [1.5], color='k', arrow_length_ratio=0.1)
    ax.text(1.6, 0, 0, 'X', fontsize=14)
    ax.text(0, 1.6, 0, 'Y', fontsize=14)
    ax.text(0, 0, 1.6, 'Z', fontsize=14)
    
    # Рисуем кубиты
    colors = ['r', 'g', 'm', 'c']
    labels = ['|ψ₀⟩', '|ψ₁⟩', '|ψ₂⟩', '|ψ₃⟩']
    for i, qubit in enumerate(qubits):
        x, y, z = qubit.get_bloch_coordinates()
        ax.quiver([0], [0], [0], [x], [y], [z], 
                 color=colors[i], arrow_length_ratio=0.1, label=labels[i])
    
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    plt.tight_layout()
    plt.show()

def print_qubit_info(qubit: Qubit, gate_name: str):
    """Выводит подробную информацию о кубите после применения гейта"""
    print(f"\nПосле гейта {gate_name}:")
    print(f"Состояние: {qubit}")
    
    # Вероятности
    prob_0 = np.abs(qubit.state[0])**2
    prob_1 = np.abs(qubit.state[1])**2
    print(f"Вероятность |0⟩: {prob_0:.3f}")
    print(f"Вероятность |1⟩: {prob_1:.3f}")
    
    # Координаты на сфере Блоха
    x, y, z = qubit.get_bloch_coordinates()
    print(f"Координаты на сфере Блоха: x={x:.3f}, y={y:.3f}, z={z:.3f}")

def demo_single_qubit_gates():
    """Демонстрация работы однокубитных гейтов с подробным выводом"""
    print("Демонстрация однокубитных гейтов")
    print("=" * 50)
    
    # Создаем кубит в состоянии |0⟩
    q = Qubit()
    print(f"\nИсходное состояние кубита: {q}")
    print_qubit_info(q, "начальное")
    
    # Применяем гейты Паули по очереди
    gates = {
        "X": QuantumSystem.pauli_x(),
        "Y": QuantumSystem.pauli_y(),
        "Z": QuantumSystem.pauli_z(),
        "H": QuantumSystem.hadamard()
    }
    
    for name, gate in gates.items():
        # Создаем копию исходного кубита для каждого гейта
        q_copy = Qubit(q.state.copy())
        q_copy.apply_gate(gate)
        
        # Выводим информацию
        print("\n" + "="*50)
        print(f"Применяем гейт {name}:")
        print(f"Матрица гейта {name}:")
        print(gate)
        print_qubit_info(q_copy, name)
        
        # Визуализация
        plot_bloch_sphere([q_copy], title=f"Состояние после гейта {name}")

def demo_cnot_gate():
    """Демонстрация работы гейта CNOT"""
    print("\nДемонстрация гейта CNOT")
    print("=" * 50)
    
    # Создаем два кубита в состоянии |00⟩
    q0 = Qubit()
    q1 = Qubit()
    
    print("\nИсходное состояние системы:")
    print(f"q0 = {q0}")
    print(f"q1 = {q1}")
    
    # Применяем гейт X к первому кубиту, чтобы получить |10⟩
    q0.apply_gate(QuantumSystem.pauli_x())
    print("\nПосле применения X к q0:")
    print(f"q0 = {q0}")
    print(f"q1 = {q1}")
    
    # Теперь применяем CNOT (q0 - управляющий, q1 - целевой)
    # Создаем состояние двух кубитов как тензорное произведение
    state = np.kron(q0.state, q1.state)
    
    # Применяем CNOT
    cnot = QuantumSystem.cnot(control=0, target=1)
    new_state = np.dot(cnot, state)
    
    # Разделяем состояние обратно на отдельные кубиты
    q0_after = Qubit(np.array([new_state[0], new_state[2]]))
    q1_after = Qubit(np.array([new_state[0], new_state[1]]))
    
    print("\nПосле применения CNOT(q0, q1):")
    print(f"q0 = {q0_after}")
    print(f"q1 = {q1_after}")
    
    # Визуализация
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': '3d'})
    
    # До CNOT
    x0, y0, z0 = q0.get_bloch_coordinates()
    x1, y1, z1 = q1.get_bloch_coordinates()
    
    for ax, title, x, y, z in [(ax1, "До CNOT", [x0, x1], [y0, y1], [z0, z1]),
                               (ax2, "После CNOT", 
                                [q0_after.get_bloch_coordinates()[0], q1_after.get_bloch_coordinates()[0]],
                                [q0_after.get_bloch_coordinates()[1], q1_after.get_bloch_coordinates()[1]],
                                [q0_after.get_bloch_coordinates()[2], q1_after.get_bloch_coordinates()[2]])]:
        
        # Рисуем сферу
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(xs, ys, zs, color='b', alpha=0.1)
        
        # Оси
        ax.quiver([0], [0], [0], [1.5], [0], [0], color='k', arrow_length_ratio=0.1)
        ax.quiver([0], [0], [0], [0], [1.5], [0], color='k', arrow_length_ratio=0.1)
        ax.quiver([0], [0], [0], [0], [0], [1.5], color='k', arrow_length_ratio=0.1)
        
        # Векторы состояний
        ax.quiver([0, 0], [0, 0], [0, 0], x, y, z, 
                 color=['r', 'g'], arrow_length_ratio=0.1, label=['q0', 'q1'])
        
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    demo_single_qubit_gates()
    demo_cnot_gate()
