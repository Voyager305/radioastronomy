#!/usr/bin/env python3
# Генерация иллюстративных графиков для гайда по радиоастрономии.
# Данные СИНТЕТИЧЕСКИЕ (не реальные наблюдения) — для объяснения понятий.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
})

F0 = 1420.406  # МГц, покоящаяся частота HI
C = 299792.458  # км/с

def gauss(x, a, mu, sig):
    return a * np.exp(-0.5 * ((x - mu) / sig) ** 2)

# ---------------------------------------------------------------------------
# 1. Профиль линии HI: ON (на Млечный Путь) vs OFF (в стороне)
# ---------------------------------------------------------------------------
def fig_hi_profile():
    f = np.linspace(F0 - 1.2, F0 + 1.2, 1400)  # МГц
    # облака водорода на разных лучевых скоростях -> несколько компонент
    profile = (gauss(f, 6.0, F0 - 0.25, 0.10)
               + gauss(f, 9.0, F0 - 0.02, 0.14)
               + gauss(f, 4.5, F0 + 0.20, 0.09))
    base = 2.0 + 0.4 * (f - F0)  # лёгкий наклон подложки
    noise = rng.normal(0, 0.35, f.size)
    on = base + profile + noise
    off = base + rng.normal(0, 0.35, f.size)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(f, off, color="#8a8f98", lw=1.1, label="OFF — в стороне от полосы")
    ax.plot(f, on, color="#1f6feb", lw=1.4, label="ON — на Млечный Путь")
    ax.axvline(F0, color="#d1242f", ls="--", lw=1, label=f"покоящаяся частота HI ({F0} МГц)")
    ax.set_xlabel("Частота, МГц")
    ax.set_ylabel("Относительная мощность")
    ax.set_title("Профиль линии водорода: сравнение ON / OFF (синтетические данные)")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    # вторая ось — лучевая скорость
    secax = ax.secondary_xaxis(
        "top",
        functions=(lambda ff: C * (F0 - ff) / F0, lambda v: F0 * (1 - v / C)),
    )
    secax.set_xlabel("Лучевая скорость, км/с")
    fig.savefig(f"{OUT}/hi-profile.png")
    plt.close(fig)

# ---------------------------------------------------------------------------
# 2. Спектр vs водопад (спектрограмма)
# ---------------------------------------------------------------------------
def fig_spectrum_waterfall():
    f = np.linspace(1419, 1422, 600)
    spec = 2 + gauss(f, 5, F0, 0.15) + rng.normal(0, 0.3, f.size)

    nt = 160
    W = rng.normal(2, 0.3, (nt, f.size))
    W += gauss(f, 4, F0, 0.12)[None, :]          # устойчивый сигнал (верт. полоса)
    W += gauss(f, 6, 1420.9, 0.02)[None, :]      # узкая помеха RFI
    # временный всплеск помехи
    W[60:70, :] += gauss(f, 5, 1419.6, 0.03)[None, :]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.plot(f, spec, color="#1f6feb", lw=1.2)
    a1.axvline(F0, color="#d1242f", ls="--", lw=1)
    a1.set_xlabel("Частота, МГц")
    a1.set_ylabel("Мощность")
    a1.set_title("Спектр (в данный момент)")

    im = a2.imshow(W, aspect="auto", origin="lower", cmap="magma",
                   extent=[f[0], f[-1], 0, nt])
    a2.set_xlabel("Частота, МГц")
    a2.set_ylabel("Время (кадры)")
    a2.set_title("Водопад (накопление во времени)")
    a2.grid(False)
    cb = fig.colorbar(im, ax=a2, fraction=0.046, pad=0.04)
    cb.set_label("Уровень")
    fig.suptitle("Одни и те же данные: слева спектр, справа спектрограмма (синтетические данные)",
                 fontsize=11)
    fig.savefig(f"{OUT}/spectrum-vs-waterfall.png")
    plt.close(fig)

# ---------------------------------------------------------------------------
# 3. Усреднение и SNR (∝ √N)
# ---------------------------------------------------------------------------
def fig_averaging():
    f = np.linspace(F0 - 1, F0 + 1, 800)
    signal = gauss(f, 1.2, F0, 0.13)  # слабая линия
    noise_sd = 1.0

    single = signal + rng.normal(0, noise_sd, f.size)
    N = 300
    stack = signal[None, :] + rng.normal(0, noise_sd, (N, f.size))
    avg = stack.mean(axis=0)

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    axs[0].plot(f, single, color="#8a8f98", lw=0.8)
    axs[0].set_title("1 спектр: линия в шуме")
    axs[1].plot(f, avg, color="#1f6feb", lw=1.3)
    axs[1].set_title(f"Усреднение {N} спектров: линия видна")
    for a in axs[:2]:
        a.axvline(F0, color="#d1242f", ls="--", lw=1)
        a.set_xlabel("Частота, МГц")
        a.set_ylabel("Мощность")
    Ns = np.arange(1, 400)
    axs[2].plot(np.sqrt(Ns), np.sqrt(Ns), color="#2da44e", lw=1.6)
    axs[2].set_xlabel("√N  (√ числа спектров)")
    axs[2].set_ylabel("Выигрыш в SNR")
    axs[2].set_title("SNR ∝ √(N·τ)")
    fig.suptitle("Слабый сигнал выделяется временем усреднения (синтетические данные)", fontsize=11)
    fig.savefig(f"{OUT}/averaging-snr.png")
    plt.close(fig)

# ---------------------------------------------------------------------------
# 4. Диаграмма «долгота–скорость» (l–v): классический радио-«снимок» HI
# ---------------------------------------------------------------------------
def fig_lv_diagram():
    lon = np.linspace(0, 180, 400)     # галактическая долгота, град
    vel = np.linspace(-150, 150, 400)  # лучевая скорость, км/с
    L, V = np.meshgrid(lon, vel)
    img = np.zeros_like(L)
    # несколько «рукавов» — синусоидальные гребни (модель галактического вращения)
    for phase, amp, wid in [(0.0, 90, 18), (1.1, 60, 14), (2.3, 70, 16)]:
        ridge = amp * np.sin(np.deg2rad(L) + phase)
        img += np.exp(-0.5 * ((V - ridge) / wid) ** 2)
    img += 0.15 * rng.normal(0, 1, img.shape) ** 2
    fig, ax = plt.subplots(figsize=(8, 4.6))
    im = ax.imshow(img, aspect="auto", origin="lower", cmap="viridis",
                   extent=[lon[0], lon[-1], vel[0], vel[-1]])
    ax.set_xlabel("Галактическая долгота (или азимут), град")
    ax.set_ylabel("Лучевая скорость, км/с")
    ax.set_title("Диаграмма «долгота–скорость» HI — радиоизображение (синтетические данные)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Интенсивность HI")
    fig.savefig(f"{OUT}/lv-diagram.png")
    plt.close(fig)

# ---------------------------------------------------------------------------
# 5. Карта яркости одиночной антенны (single-dish): полоса Млечного Пути
# ---------------------------------------------------------------------------
def fig_single_dish_map():
    az = np.linspace(60, 300, 240)   # азимут, град
    el = np.linspace(5, 80, 180)     # высота, град
    A, E = np.meshgrid(az, el)
    band = 45 + 12 * np.sin(np.deg2rad(A - 60) * 1.2)  # «полоса» Галактики
    img = np.exp(-0.5 * ((E - band) / 9) ** 2)
    img += 0.5 * np.exp(-0.5 * (((A - 150) / 12) ** 2 + ((E - 40) / 8) ** 2))  # яркая область
    img += 0.08 * rng.normal(0, 1, img.shape) ** 2
    fig, ax = plt.subplots(figsize=(8, 4.4))
    im = ax.imshow(img, aspect="auto", origin="lower", cmap="inferno",
                   extent=[az[0], az[-1], el[0], el[-1]])
    ax.set_xlabel("Азимут, град")
    ax.set_ylabel("Высота, град")
    ax.set_title("Карта яркости HI одиночной антенной (растровое сканирование, синтетические данные)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Интенсивность")
    fig.savefig(f"{OUT}/single-dish-map.png")
    plt.close(fig)

# ---------------------------------------------------------------------------
# 6. Доплеровский сдвиг профиля по направлениям
# ---------------------------------------------------------------------------
def fig_doppler():
    v = np.linspace(-200, 200, 900)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for shift, col, lab in [(-40, "#1f6feb", "направление A"),
                            (10, "#2da44e", "направление B"),
                            (60, "#d1242f", "направление C")]:
        prof = gauss(v, 1.0, shift, 22) + gauss(v, 0.5, shift + 45, 15)
        ax.plot(v, prof + rng.normal(0, 0.02, v.size), color=col, lw=1.5, label=lab)
    ax.axvline(0, color="#57606a", ls="--", lw=1, label="покоящаяся частота (v=0)")
    ax.set_xlabel("Лучевая скорость, км/с")
    ax.set_ylabel("Относительная интенсивность")
    ax.set_title("Смещение профиля HI по направлениям — эффект Доплера (синтетические данные)")
    ax.legend(fontsize=9, framealpha=0.9)
    fig.savefig(f"{OUT}/doppler-shift.png")
    plt.close(fig)

for fn in (fig_hi_profile, fig_spectrum_waterfall, fig_averaging,
           fig_lv_diagram, fig_single_dish_map, fig_doppler):
    fn()
    print("создан:", fn.__name__)

print("\nФайлы в", OUT)
for p in sorted(os.listdir(OUT)):
    print(" ", p)
