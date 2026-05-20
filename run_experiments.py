import numpy as np
import os
import time

# ============================================================
# 1. ГЕНЕРАЦИЯ ПАТТЕРНОВ И ОБУЧЕНИЕ (Раздел 1)
# ============================================================
def generate_patterns(P, N):
    """Генерирует P случайных бинарных паттернов длины N."""
    return np.random.choice([-1, 1], size=(P, N))

def hebbian_learning(patterns, N):
    """Вычисляет весовую матрицу по правилу Хебба."""
    W = (patterns.T @ patterns) / N
    np.fill_diagonal(W, 0.0)
    return W

# ============================================================
# 2. АСИНХРОННАЯ ДИНАМИКА И КРИТЕРИИ (Раздел 2)
# ============================================================
def async_update(W, s, max_iter=500):
    """
    Асинхронное обновление состояний нейронов.
    Нейроны обновляются по одному в случайном порядке до сходимости.
    """
    N = len(s)
    s = s.copy()
    for _ in range(max_iter):
        order = np.random.permutation(N)
        changed = False
        for i in order:
            h_i = np.dot(W[i], s)  # W[i,i] == 0, поэтому самовлияние исключено
            s_i_new = 1 if h_i >= 0 else -1
            if s_i_new != s[i]:
                s[i] = s_i_new
                changed = True
        if not changed:
            break
    return s

def get_min_hamming(s_final, patterns):
    """Возвращает минимальное расстояние Хэмминга до любого паттерна или его инверсии."""
    dists = np.sum(s_final != patterns, axis=1)
    dists_inv = np.sum(s_final != -patterns, axis=1)
    return np.min(np.concatenate([dists, dists_inv]))

# ============================================================
# 3. ПРОЦЕДУРА РАЗ-ОБУЧЕНИЯ / «СОН» (Раздел 3)
# ============================================================
def unlearn_step(W, eta):
    """Один шаг анти-хеббиановского раз-обучения."""
    N = W.shape[0]
    s_init = np.random.choice([-1, 1], size=N)
    s_att = async_update(W, s_init)
    W -= (eta / N) * np.outer(s_att, s_att)
    np.fill_diagonal(W, 0.0)
    return W

def run_unlearning(W, n_steps, eta):
    """Запускает процедуру «сна» на n_steps эпох."""
    W_sleep = W.copy()
    for _ in range(n_steps):
        W_sleep = unlearn_step(W_sleep, eta)
    return W_sleep

# ============================================================
# 4. МЕТРИКИ И ЭКСПЕРИМЕНТ (Раздел 4)
# ============================================================
def evaluate_recovery_and_spurious(W, patterns, noise_frac=0.1, n_trials=1000):
    """Оценивает долю успешных восстановлений и частоту спурионных состояний."""
    N, P = W.shape[0], patterns.shape[0]
    success_count = 0
    spurious_count = 0

    for _ in range(n_trials):
        idx = np.random.randint(P)
        s = patterns[idx].copy()
        # Добавление шума
        flip_mask = np.random.rand(N) < noise_frac
        s[flip_mask] *= -1

        s_final = async_update(W, s)
        min_dist = get_min_hamming(s_final, patterns)

        if min_dist <= 0.1 * N:
            success_count += 1
        elif min_dist > 0.15 * N:
            spurious_count += 1

    return success_count / n_trials, spurious_count / n_trials

def evaluate_basins(W, patterns, n_trials=5000):
    """Оценивает размеры бассейнов притяжения для каждого паттерна."""
    N, P = W.shape[0], patterns.shape[0]
    basin_sizes = np.zeros(P)

    for _ in range(n_trials):
        s_init = np.random.choice([-1, 1], size=N)
        s_final = async_update(W, s_init)
        min_dist = get_min_hamming(s_final, patterns)
        # Находим индекс ближайшего паттерна (или инверсии)
        dists = np.sum(s_final != patterns, axis=1)
        dists_inv = np.sum(s_final != -patterns, axis=1)
        closest_idx = np.argmin(np.minimum(dists, dists_inv))
        basin_sizes[closest_idx] += 1

    return basin_sizes / n_trials

def run_full_experiment():
    # Фиксация seed для полной воспроизводимости
    np.random.seed(42)

    N = 100
    alphas = np.arange(0.05, 0.19, 0.02)
    noise_frac = 0.1
    n_trials_eval = 1000
    n_trials_basin = 5000

    etas = np.linspace(0.01, 0.10, 5)  # [0.01, 0.0325, 0.055, 0.0775, 0.10]
    sleep_steps_list = [100, 500, 1000, 2000]

    results = {}

    print("Начало вычислительного эксперимента...")
    t_start = time.time()

    for alpha in alphas:
        P = int(alpha * N)
        patterns = generate_patterns(P, N)
        W = hebbian_learning(patterns, N)

        print(f"\nAlpha={alpha:.2f} (P={P}) | Обучение завершено. Оценка без сна...")
        rec_rate_0, spur_rate_0 = evaluate_recovery_and_spurious(W, patterns, noise_frac, n_trials_eval)
        basins_0 = evaluate_basins(W, patterns, n_trials_basin)

        results[(alpha, 0, 0.0)] = {
            'rec': rec_rate_0, 'spur': spur_rate_0, 'basins': basins_0
        }

        for eta in etas:
            for n_steps in sleep_steps_list:
                print(f"  -> Сон: eta={eta:.3f}, steps={n_steps}")
                W_sleep = run_unlearning(W, n_steps, eta)
                rec, spur = evaluate_recovery_and_spurious(W_sleep, patterns, noise_frac, n_trials_eval)
                basins = evaluate_basins(W_sleep, patterns, n_trials_basin)

                results[(alpha, n_steps, eta)] = {
                    'rec': rec, 'spur': spur, 'basins': basins
                }

    # Сохранение результатов
    os.makedirs('data', exist_ok=True)
    np.savez('data/results.npz',
             alphas=alphas, etas=etas, steps=np.array(sleep_steps_list),
             results=results, metadata={'N': N, 'noise': noise_frac})

    t_end = time.time()
    print(f"\nЭксперимент завершён за {t_end - t_start:.1f} сек. Результаты сохранены в data/results.npz")

if __name__ == "__main__":
    run_full_experiment()
