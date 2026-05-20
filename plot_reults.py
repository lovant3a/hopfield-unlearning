import numpy as np
import matplotlib.pyplot as plt
import os

def load_results():
    data = np.load('data/results.npz', allow_pickle=True)
    return data['alphas'], data['etas'], data['steps'], data['results'].item()

def find_key(results, alpha, steps, eta):
    """Ищет ключ, близкий к (alpha, steps, eta), с учётом погрешности float."""
    for key in results.keys():
        if isinstance(key, tuple) and len(key) == 3:
            a, s, e = key
            if abs(float(a) - float(alpha)) < 1e-10 and int(s) == int(steps) and abs(float(e) - float(eta)) < 1e-10:
                return key
    return None

def plot_recovery_vs_alpha(alphas, results, eta=0.055, steps=1000):
    """График доли успешных восстановлений от нагрузки сети."""
    rec_before = []
    rec_after = []

    for a in alphas:
        # Для "до" раз-обучения
        key_before = find_key(results, a, 0, 0.0)
        if key_before:
            rec_before.append(results[key_before]['rec'])
        else:
            rec_before.append(np.nan)

        # Для "после" раз-обучения
        key_after = find_key(results, a, steps, eta)
        if key_after:
            rec_after.append(results[key_after]['rec'])
        else:
            rec_after.append(np.nan)

    plt.figure(figsize=(8, 5))
    plt.plot(alphas, rec_before, 'o-', label='Без раз-обучения', linewidth=2, markersize=8)
    plt.plot(alphas, rec_after, 's--', label=f'После сна (η={eta}, T={steps})', linewidth=2, markersize=8)
    plt.xlabel('Загрузка сети $\\alpha = P/N$', fontsize=12)
    plt.ylabel('Доля успешных восстановлений', fontsize=12)
    plt.title('Влияние раз-обучения на качество восстановления памяти', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('figs/rec_vs_alpha.png', dpi=300)
    plt.show()

def plot_spurious_vs_alpha(alphas, results, eta=0.055, steps=1000):
    """График частоты спурионных состояний от нагрузки."""
    spur_before = []
    spur_after = []

    for a in alphas:
        key_before = find_key(results, a, 0, 0.0)
        if key_before:
            spur_before.append(results[key_before]['spur'])
        else:
            spur_before.append(np.nan)

        key_after = find_key(results, a, steps, eta)
        if key_after:
            spur_after.append(results[key_after]['spur'])
        else:
            spur_after.append(np.nan)

    plt.figure(figsize=(8, 5))
    plt.plot(alphas, spur_before, 'o-', color='tab:red', label='Без раз-обучения', linewidth=2)
    plt.plot(alphas, spur_after, 's--', color='tab:green', label=f'После сна (η={eta}, T={steps})', linewidth=2)
    plt.xlabel('Загрузка сети $\\alpha = P/N$', fontsize=12)
    plt.ylabel('Частота спурионных аттракторов', fontsize=12)
    plt.title('Подавление ложных минимумов энергии', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('figs/spur_vs_alpha.png', dpi=300)
    plt.show()

def plot_heatmap_unlearning(alphas, etas, steps_list, results):
    """Тепловая карта: доля восстановления от η и числа шагов (фиксированный α=0.11)."""
    alpha_fix = 0.11
    Z = np.zeros((len(steps_list), len(etas)))

    for i, st in enumerate(steps_list):
        for j, et in enumerate(etas):
            key = find_key(results, alpha_fix, st, et)
            if key:
                Z[i, j] = results[key]['rec']
            else:
                Z[i, j] = np.nan

    plt.figure(figsize=(8, 5))
    im = plt.imshow(Z, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(im, label='Доля восстановления')
    plt.xticks(np.arange(len(etas)), [f"{e:.2f}" for e in etas])
    plt.yticks(np.arange(len(steps_list)), [str(s) for s in steps_list])
    plt.xlabel('Коэффициент раз-обучения $\\eta$')
    plt.ylabel('Число шагов «сна»')
    plt.title(f'Зависимость качества восстановления от гиперпараметров ($\\alpha={alpha_fix}$)')
    plt.tight_layout()
    plt.savefig('figs/unlearning_heatmap.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    os.makedirs('figs', exist_ok=True)
    alphas, etas, steps, results = load_results()

    print("Построение графиков...")

    # ИЗМЕНЕНО: берём мягкие параметры из успешной зоны тепловой карты
    eta_good = 0.01
    steps_good = 100

    plot_recovery_vs_alpha(alphas, results, eta=eta_good, steps=steps_good)
    plot_spurious_vs_alpha(alphas, results, eta=eta_good, steps=steps_good)

    # Тепловую карту оставляем как есть, она отличная!
    plot_heatmap_unlearning(alphas, etas, steps, results)

    print("Графики сохранены в папке figs/")
