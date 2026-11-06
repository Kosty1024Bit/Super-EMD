"""Итеративная фильтрация равномерным ядром и анализ мгновенной частоты.

Это kernel-based IF/RIF из статьи, а не сплайновая реализация классического EMD.
NumPy и SciPy; длина относится к элементарному вектору, до автокорреляции.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import warnings

import numpy as np
from scipy import ndimage, signal
from scipy.integrate import cumulative_trapezoid


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name}: требуется целое число")
    if value < minimum:
        raise ValueError(f"{name} должно быть >= {minimum}")
    return int(value)


def _real_signal(x):
    if np.iscomplexobj(x):
        raise ValueError("Входной сигнал должен быть вещественным")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or len(x) < 3 or not np.all(np.isfinite(x)):
        raise ValueError("Нужен конечный одномерный сигнал минимум из 3 отсчётов")
    return x


def _sample_rate(fs):
    if not np.isscalar(fs) or not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs должна быть конечной положительной частотой в Гц")
    return float(fs)


@dataclass
class SiftResult:
    mode: np.ndarray
    iterations: int
    length: int | np.ndarray
    reason: str
    mean_ratio: float


@dataclass
class Decomposition:
    modes: np.ndarray
    residual: np.ndarray
    lengths: list[int]
    iterations: list[int]
    reason: str

    def reconstruct(self):
        return self.residual + self.modes.sum(axis=0)


@dataclass
class AnalyticFeatures:
    z: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    frequency: np.ndarray
    oscillator: np.ndarray
    valid: np.ndarray


# %% Центрированный фильтр и фиксированный масштаб: формулы (2), (5), (13), (15).
def averaging_kernel(length, *, double=True):
    """Возвращает веса и целочисленные смещения центрированного фильтра.

    double=True: w=b*reverse(b), длина 2L-1, возможно любое целое L.
    double=False: сам b с весами 1/L; для центрирования нужна нечётная L.
    """
    length = _integer(length, "length")
    if not double and length % 2 == 0:
        raise ValueError("Для одинарного центрированного фильтра нужна нечётная L")
    b = np.ones(length, dtype=np.float64) / length
    weights = np.convolve(b, b[::-1]) if double else b
    radius = len(weights) // 2
    return weights, np.arange(-radius, radius + 1)


def periodic_response(n, length, *, double=True):
    """ДПФ именно центрированного эффективного фильтра, без фазового сдвига."""
    n = _integer(n, "n", 3)
    weights, offsets = averaging_kernel(length, double=double)
    impulse = np.zeros(n)
    np.add.at(impulse, offsets % n, weights)
    response = np.fft.rfft(impulse).real
    return np.clip(response, 0.0, 1.0) if double else response


def moving_average(x, length, *, double=True, boundary="reflect"):
    """Зеркальное (half-sample reflect) или циклическое продолжение."""
    x = _real_signal(x)
    if boundary not in {"reflect", "wrap"}:
        raise ValueError("boundary: 'reflect' или 'wrap'")
    weights, _ = averaging_kernel(length, double=double)
    return ndimage.convolve1d(x, weights, mode=boundary, origin=0)


def sift_fixed(x, length, *, iterations=8, double=True, boundary="reflect",
               backend="auto", mean_tol=None, min_iterations=1):
    """h <- h-Mh; фиксированный фильтр внутри одного просеивания.

    iterations — конечное число шагов / верхний предел при mean_tol.
    mean_tol: ||Mh||/||h||; это критерий среднего по ядру, не гарантия IMF.
    FFT допустима только для wrap и фиксированного числа шагов.
    """
    x = _real_signal(x)
    iterations = _integer(iterations, "iterations", 0)
    min_iterations = _integer(min_iterations, "min_iterations", 0)
    length = _integer(length, "length")
    if boundary not in {"reflect", "wrap"} or backend not in {"auto", "fft", "direct"}:
        raise ValueError("Неверные boundary или backend")
    if mean_tol is not None and (not np.isfinite(mean_tol) or mean_tol <= 0):
        raise ValueError("mean_tol должна быть положительной")
    averaging_kernel(length, double=double)
    if backend == "fft" and (boundary != "wrap" or mean_tol is not None):
        raise ValueError("FFT требует wrap и mean_tol=None")
    use_fft = backend == "fft" or (backend == "auto" and boundary == "wrap" and mean_tol is None)
    if use_fft:
        response = periodic_response(len(x), length, double=double)
        h = np.fft.irfft(np.fft.rfft(x) * (1.0 - response)**iterations, n=len(x))
        count, reason = iterations, "fixed_iterations"
    else:
        h, count, reason = x.copy(), 0, "fixed_iterations"
        for k in range(iterations):
            mean = moving_average(h, length, double=double, boundary=boundary)
            ratio = np.linalg.norm(mean) / max(np.linalg.norm(h), np.finfo(float).tiny)
            if mean_tol is not None and k >= min_iterations and ratio <= mean_tol:
                reason = "mean_tolerance"
                break
            h = h - mean
            count += 1
            if not np.all(np.isfinite(h)):
                raise FloatingPointError("Просеивание стало нечисловым")
    mean = moving_average(h, length, double=double, boundary=boundary)
    ratio = float(np.linalg.norm(mean) / max(np.linalg.norm(h), np.finfo(float).tiny))
    return SiftResult(h, count, length, reason, ratio)


def decompose_fixed(x, lengths, *, iterations=8, boundary="wrap", backend="auto"):
    """Последовательное извлечение: c_j=P_j r_(j-1); остаток сохраняется."""
    x = _real_signal(x)
    lengths = list(lengths)
    counts = [iterations] * len(lengths) if isinstance(iterations, Integral) else list(iterations)
    if len(counts) != len(lengths):
        raise ValueError("Нужен один iterations на каждую длину")
    modes, used, residual = [], [], x.copy()
    for length, count in zip(lengths, counts):
        result = sift_fixed(residual, length, iterations=count,
                            boundary=boundary, backend=backend, double=True)
        modes.append(result.mode)
        used.append(result.iterations)
        residual = residual - result.mode
    modes = np.stack(modes) if modes else np.empty((0, len(x)))
    return Decomposition(modes, residual, [int(v) for v in lengths], used, "given_lengths")


def estimate_length_extrema(x):
    """Начальная эвристика: медиана полных периодов между однотипными экстремумами.

    Полный период уменьшает ошибку округления двух отдельных полупериодов.
    Ненадёжна при дополнительных экстремумах от шума и смешении компонент.
    """
    x = _real_signal(x)
    peaks = signal.find_peaks(x)[0]
    troughs = signal.find_peaks(-x)[0]
    periods = np.r_[np.diff(peaks), np.diff(troughs)]
    if len(periods) < 2:
        raise ValueError("Недостаточно экстремумов для оценки периода")
    return int(np.clip(np.rint(np.median(periods)), 3, len(x)))


def decompose_auto(x, fs, *, max_modes=6, iterations=8, boundary="reflect",
                   residual_tol=1e-8, mode_tol=1e-8, mean_tol=None):
    """Выбор L по текущему остатку; это эвристика без универсальной гарантии."""
    x, fs = _real_signal(x), _sample_rate(fs)
    max_modes = _integer(max_modes, "max_modes", 0)
    if residual_tol < 0 or mode_tol < 0:
        raise ValueError("Допуски должны быть неотрицательными")
    initial_norm = np.linalg.norm(x)
    residual, modes, lengths, counts = x.copy(), [], [], []
    reason = "max_modes"
    if initial_norm == 0:
        reason = "zero_signal"
    else:
        for _ in range(max_modes):
            if np.linalg.norm(residual) <= residual_tol * initial_norm:
                reason = "small_residual"
                break
            try:
                length = estimate_length_extrema(residual)
            except ValueError:
                reason = "insufficient_extrema"
                break
            result = sift_fixed(residual, length, iterations=iterations, boundary=boundary,
                                mean_tol=mean_tol, double=True)
            if np.linalg.norm(result.mode) <= mode_tol * initial_norm:
                reason = "small_mode"
                break
            modes.append(result.mode)
            lengths.append(length)
            counts.append(result.iterations)
            residual = residual - result.mode
    modes = np.stack(modes) if modes else np.empty((0, len(x)))
    return Decomposition(modes, residual, lengths, counts, reason)


# %% Переменная длина и фазовая передискретизация: формулы (10), (21)-(23).
def moving_average_variable(x, lengths, *, boundary="reflect"):
    """Точное локальное усреднение по нечётным L[n], через префиксные суммы."""
    x = _real_signal(x)
    lengths = np.asarray(lengths)
    if lengths.shape != x.shape or lengths.dtype.kind not in "iu":
        raise ValueError("lengths: целочисленный профиль длины сигнала")
    if np.any(lengths < 1) or np.any(lengths % 2 == 0) or np.any(lengths > len(x)):
        raise ValueError("Все L[n] должны быть положительными, нечётными и <= N")
    if boundary not in {"reflect", "wrap"}:
        raise ValueError("boundary: 'reflect' или 'wrap'")
    radii = (lengths - 1) // 2
    padding = int(radii.max())
    extended = np.pad(x, padding, mode="symmetric" if boundary == "reflect" else "wrap")
    prefix = np.r_[0.0, np.cumsum(extended)]
    centers = padding + np.arange(len(x))
    return (prefix[centers + radii + 1] - prefix[centers - radii]) / lengths


def sift_variable(x, lengths, *, iterations=8, boundary="reflect", growth_limit=5.0):
    """Экспериментальный ALIF с одинарным средним и замороженным L[n].

    Не переносит доказательство устойчивости фиксированной автокорреляции.
    Превышение роста вызывает исключение; это контроль, а не доказательство.
    """
    x = _real_signal(x)
    iterations = _integer(iterations, "iterations", 0)
    if not np.isfinite(growth_limit) or growth_limit <= 1:
        raise ValueError("growth_limit должен быть конечным и > 1")
    norm0, h = np.linalg.norm(x), x.copy()
    for _ in range(iterations):
        h = h - moving_average_variable(h, lengths, boundary=boundary)
        if not np.all(np.isfinite(h)) or np.linalg.norm(h) > growth_limit * max(norm0, np.finfo(float).tiny):
            raise RuntimeError("Переменный фильтр превысил допустимый рост")
    mean = moving_average_variable(h, lengths, boundary=boundary)
    ratio = np.linalg.norm(mean) / max(np.linalg.norm(h), np.finfo(float).tiny)
    return SiftResult(h, iterations, np.asarray(lengths).copy(), "fixed_iterations", float(ratio))


def phase_resampled_sift(x, fs, frequency_guide, *, iterations=8,
                          samples_per_cycle=64, oversample=4, max_grid_points=2_000_000):
    """Фильтрация в v=integral f_hat dt. Линейная интерполяция, без сплайнов.

    FIR-передискретизация во времени уменьшает интерполяционную ошибку.
    Элементарный вектор имеет равные веса на фазовой, а не временной сетке.
    Возвращает моду и диагностическую информацию о сетке.
    """
    x, fs = _real_signal(x), _sample_rate(fs)
    guide = np.asarray(frequency_guide, dtype=float)
    if guide.shape != x.shape or not np.all(np.isfinite(guide)):
        raise ValueError("frequency_guide должен быть конечным массивом длины N")
    if np.any(guide <= 0) or np.any(guide >= fs / 2):
        raise ValueError("Нужна положительная частота строго ниже Найквиста")
    samples_per_cycle = _integer(samples_per_cycle, "samples_per_cycle", 8)
    oversample = _integer(oversample, "oversample")
    max_grid_points = _integer(max_grid_points, "max_grid_points", 3)
    t = np.arange(len(x)) / fs
    if oversample > 1:
        x_up = signal.resample_poly(x, oversample, 1, padtype="reflect")[:oversample * (len(x) - 1) + 1]
        t_up = np.arange(len(x_up)) / (fs * oversample)
        f_up = np.interp(t_up, t, guide)
    else:
        x_up, t_up, f_up = x, t, guide
    v_up = cumulative_trapezoid(f_up, t_up, initial=0.0)
    if v_up[-1] < 3:
        raise ValueError("Нужно минимум три периода для фазового просеивания")
    grid_size = int(round(v_up[-1] * samples_per_cycle)) + 1
    if grid_size > max_grid_points:
        raise ValueError("Фазовая сетка слишком велика: уменьшите samples_per_cycle")
    v_grid = np.linspace(0, v_up[-1], grid_size)
    dv = v_grid[1] - v_grid[0]
    length = int(round(1.0 / dv))
    warped = np.interp(v_grid, v_up, x_up)
    result = sift_fixed(warped, length, iterations=iterations, boundary="reflect", double=True)
    v_at_t = np.interp(t, t_up, v_up)
    mode = np.interp(v_at_t, v_grid, result.mode)
    info = {"phase_grid_points": grid_size, "length_in_phase": length,
            "cycle_window": length * dv, "oversample": oversample,
            "mean_ratio": result.mean_ratio}
    return mode, info


# %% Аналитический сигнал, квадратура, осцилляторы: формулы (16)-(18), (24).
def analytic_features(x, fs, *, boundary="wrap", trim=0, amplitude_floor=1e-3,
                       smooth_window=None, derivative="phase"):
    """Частота в Гц; недостоверные отсчёты получают NaN, знак не исправляется.

    derivative='phase': центрированная разность развёрнутой фазы.
    derivative='spectral': Im(z'/z)/(2pi), только для периодических данных.
    smooth_window: нечётное окно Savitzky-Golay для производной фазы.
    """
    x, fs = _real_signal(x), _sample_rate(fs)
    trim = _integer(trim, "trim", 0)
    if 2 * trim >= len(x):
        raise ValueError("trim удаляет весь сигнал")
    if not 0 <= amplitude_floor < 1:
        raise ValueError("amplitude_floor должен лежать в [0,1)")
    if boundary not in {"wrap", "reflect"}:
        raise ValueError("boundary: 'reflect' или 'wrap'")
    if derivative not in {"phase", "spectral"}:
        raise ValueError("derivative: 'phase' или 'spectral'")
    if derivative == "spectral" and (boundary != "wrap" or smooth_window is not None):
        raise ValueError("Спектральная производная требует wrap без smooth_window")
    if boundary == "reflect":
        padding = min(len(x) - 1, max(3, int(fs)))
        z = signal.hilbert(np.pad(x, padding, mode="symmetric"))[padding:padding + len(x)]
    else:
        z = signal.hilbert(x)
    amplitude, phase = np.abs(z), np.unwrap(np.angle(z))
    width = 3
    if derivative == "spectral":
        dz = np.fft.ifft(2j * np.pi * np.fft.fftfreq(len(z), 1 / fs) * np.fft.fft(z))
        frequency = np.full(len(x), np.nan)
        np.divide(dz, z, out=dz, where=amplitude > 0)
        frequency[amplitude > 0] = dz[amplitude > 0].imag / (2 * np.pi)
    elif smooth_window is None:
        frequency = np.gradient(phase, 1 / fs, edge_order=2) / (2 * np.pi)
    else:
        width = _integer(smooth_window, "smooth_window", 5)
        if width % 2 == 0 or width > len(x):
            raise ValueError("smooth_window должна быть нечётной и <= N")
        frequency = signal.savgol_filter(phase, width, 3, deriv=1, delta=1 / fs) / (2 * np.pi)
    threshold = amplitude_floor * amplitude.max()
    valid = amplitude > threshold
    # Дефект фазы влияет и на соседние отсчёты дифференцирования.
    valid = ndimage.minimum_filter1d(valid.astype(np.uint8), size=width, mode="constant", cval=0).astype(bool)
    if trim:
        valid[:trim] = False
        valid[-trim:] = False
    frequency[~valid] = np.nan
    oscillator = np.full(len(x), np.nan + 1j * np.nan)
    np.divide(z, amplitude, out=oscillator, where=amplitude > 0)
    return AnalyticFeatures(z, amplitude, phase, frequency, oscillator, valid)


def quadrature_features(x, fs, envelope, *, trim=2):
    """Прямая квадратура при положительной известной/оценённой огибающей.

    Предполагаются монотонная возрастающая фаза и |x/envelope|<=1.
    Выход за диапазон не скрывается обрезанием больших ошибок.
    """
    x, fs = _real_signal(x), _sample_rate(fs)
    envelope = np.broadcast_to(np.asarray(envelope, dtype=float), x.shape)
    if not np.all(np.isfinite(envelope)) or np.any(envelope <= 0):
        raise ValueError("Огибающая должна быть конечной и строго положительной")
    v = x / envelope
    if np.any(np.abs(v) > 1 + 1e-10):
        raise ValueError("|x/envelope|>1: огибающая не согласована с сигналом")
    v = np.clip(v, -1, 1)
    sign = np.sign(np.gradient(v, 1 / fs, edge_order=2))
    nonzero = np.flatnonzero(sign)
    if not len(nonzero):
        raise ValueError("Не определено направление фазы постоянного сигнала")
    latest = np.maximum.accumulate(np.where(sign != 0, np.arange(len(v)), nonzero[0]))
    sign = sign[latest]
    q = -sign * np.sqrt(np.maximum(0, 1 - v**2))
    z = envelope * (v + 1j * q)
    phase = np.unwrap(np.angle(z))
    frequency = np.gradient(phase, 1 / fs, edge_order=2) / (2 * np.pi)
    trim = _integer(trim, "trim", 0)
    if 2 * trim >= len(x):
        raise ValueError("trim удаляет весь сигнал")
    valid = np.ones(len(x), dtype=bool)
    if trim:
        valid[:trim] = valid[-trim:] = False
    frequency[~valid] = np.nan
    return AnalyticFeatures(z, np.abs(z), phase, frequency, z / np.abs(z), valid)


def hilbert_guided_extract(x, fs, *, initial_length=None, iterations=8,
                            refinements=2, trim_seconds=1.0, smooth_window=51,
                            samples_per_cycle=64, oversample=4):
    """Начальный масштаб по экстремумам -> пилот -> фаза -> RIF.

    Истинные частоты не принимаются этой функцией. Профиль фиксируется
    внутри каждого просеивания; уточнение выполняется между его запусками.
    """
    x, fs = _real_signal(x), _sample_rate(fs)
    refinements = _integer(refinements, "refinements")
    if not np.isfinite(trim_seconds) or trim_seconds < 0:
        raise ValueError("trim_seconds должна быть неотрицательной")
    length = estimate_length_extrema(x) if initial_length is None else _integer(initial_length, "initial_length")
    pilot = sift_fixed(x, length, iterations=iterations, boundary="reflect").mode
    history, guide = [], None
    for _ in range(refinements):
        features = analytic_features(pilot, fs, boundary="reflect", trim=int(round(trim_seconds * fs)),
                                     amplitude_floor=0.05, smooth_window=smooth_window)
        valid = features.valid & (features.frequency > 0) & (features.frequency < fs / 2)
        interior = np.arange(len(x))
        trimmed = (interior >= int(round(trim_seconds * fs))) & (interior < len(x) - int(round(trim_seconds * fs)))
        if valid.sum() < 20 or valid[trimmed].mean() < 0.8:
            raise ValueError("Недостаточно надёжных положительных фазовых оценок")
        # Заполнение только для фильтрации; при оценке качества края исключаются.
        guide = np.interp(interior, interior[valid], features.frequency[valid])
        pilot, info = phase_resampled_sift(x, fs, guide, iterations=iterations,
                                           samples_per_cycle=samples_per_cycle, oversample=oversample)
        history.append(info | {"valid_fraction": float(valid.mean())})
    return pilot, guide, {"initial_length": length, "history": history}


def synthesize_from_intervals(amplitudes, phase0, interval_frequencies, residual, fs):
    """Дискретный вариант (24): f[n+1/2] — средняя частота на интервале.

    Интеграл = накопленная сумма f*dt; phase0 сохраняется, как у ДПФ.
    Для дифференцированной точечной частоты квадратура лишь приближённая.
    """
    fs, residual = _sample_rate(fs), _real_signal(residual)
    amplitudes = np.asarray(amplitudes, dtype=float)
    interval_frequencies = np.asarray(interval_frequencies, dtype=float)
    phase0 = np.asarray(phase0, dtype=float)
    if amplitudes.ndim != 2 or amplitudes.shape[1] != len(residual):
        raise ValueError("amplitudes: массив J x N")
    j, n = amplitudes.shape
    if phase0.shape != (j,) or interval_frequencies.shape != (j, n - 1):
        raise ValueError("Неверные размеры phase0 или interval_frequencies")
    if not all(np.all(np.isfinite(v)) for v in [amplitudes, phase0, interval_frequencies]):
        raise ValueError("Синтез требует конечных амплитуд и фаз на данном интервале")
    phase = phase0[:, None] + np.c_[np.zeros(j), 2 * np.pi * np.cumsum(interval_frequencies, axis=1) / fs]
    return residual + np.sum(amplitudes * np.cos(phase), axis=0)


# %% Условные интегральные операторы и локальная проекция: (8), (9), (25), (28).
def periodic_average_matrix(n, length, *, double=True):
    """Массы дискретного интегрального ядра. Только для небольших проверок."""
    n = _integer(n, "n", 3)
    if n > 512:
        raise ValueError("Плотная матрица предназначена только для N<=512")
    weights, offsets = averaging_kernel(length, double=double)
    matrix = np.zeros((n, n))
    rows = np.arange(n)
    for weight, offset in zip(weights, offsets):
        matrix[rows, (rows - offset) % n] += weight
    return matrix


def decomposition_operators(n, lengths, iterations=8):
    """K_j=P_j R_(j-1), R_j=(I-P_j) R_(j-1); параметры уже зафиксированы."""
    n = _integer(n, "n", 3)
    iterations = _integer(iterations, "iterations", 0)
    if n > 512:
        raise ValueError("Операторы предназначены только для N<=512")
    identity, remainder = np.eye(n), np.eye(n)
    kernels = []
    for length in lengths:
        mean = periodic_average_matrix(n, length)
        p = np.linalg.matrix_power(identity - mean, iterations)
        kernels.append(p @ remainder)
        remainder = (identity - p) @ remainder
    return kernels, remainder


def oscillator_projection(z, oscillators, *, window=None, ridge=0.0):
    """Взвешенная комплексная проекция с поправкой на матрицу Грама."""
    z = np.asarray(z, dtype=complex)
    u = np.asarray(oscillators, dtype=complex)
    if z.ndim != 1 or u.ndim != 2 or u.shape[1] != len(z) or not len(u):
        raise ValueError("z: N; oscillators: J x N")
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(u)):
        raise ValueError("Проекция требует конечных значений")
    weights = np.ones(len(z)) if window is None else np.asarray(window, dtype=float)
    if weights.shape != z.shape or not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Окно должно быть неотрицательным, длины N, с положительной суммой")
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge должна быть конечной и неотрицательной")
    weights = weights / weights.sum()
    gram = (u.conj() * weights) @ u.T
    b = (u.conj() * weights) @ z
    condition = float(np.linalg.cond(gram))
    if condition > 1e8 and ridge == 0:
        warnings.warn("Матрица Грама плохо обусловлена; коэффициенты ненадёжны", RuntimeWarning, stacklevel=2)
    coefficients = np.linalg.lstsq(gram + ridge * np.eye(len(u)), b, rcond=None)[0]
    return coefficients, gram, condition


def relative_error(actual, reference, mask=None):
    """Относительная L2-ошибка; при нулевой эталонной норме не определена."""
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.shape != reference.shape:
        raise ValueError("Размеры actual и reference должны совпадать")
    if mask is not None:
        actual, reference = actual[mask], reference[mask]
    denominator = np.linalg.norm(reference)
    if denominator == 0:
        return float("nan")
    return float(np.linalg.norm(actual - reference) / denominator)


def frequency_density(features, frequency_grid, *, keep_phase=True):
    """Биннированный вариант (27) либо амплитудный спектр Гильберта.

    Плотность делится на df: интеграл по частоте = df*sum(density).
    Неопределённые и выходящие за сетку значения сохраняются в unassigned.
    Это необходимо для точного синтеза при keep_phase=True.
    """
    features = list(features)
    grid = np.asarray(frequency_grid, dtype=float)
    if not features or grid.ndim != 1 or len(grid) < 2 or not np.all(np.isfinite(grid)):
        raise ValueError("Нужны компоненты и конечная частотная сетка")
    df = grid[1] - grid[0]
    if df <= 0 or not np.allclose(np.diff(grid), df):
        raise ValueError("Сетка должна быть равномерной и возрастающей")
    n = len(features[0].z)
    dtype = complex if keep_phase else float
    density, unassigned = np.zeros((len(grid), n), dtype=dtype), np.zeros(n, dtype=dtype)
    for feature in features:
        if len(feature.z) != n:
            raise ValueError("Компоненты должны иметь одинаковую длину")
        weights = feature.z if keep_phase else feature.amplitude
        valid = feature.valid & np.isfinite(feature.frequency) & (feature.frequency >= grid[0]) & (feature.frequency <= grid[-1])
        samples = np.flatnonzero(valid)
        bins = np.rint((feature.frequency[valid] - grid[0]) / df).astype(int)
        np.add.at(density, (bins, samples), weights[valid] / df)
        unassigned[~valid] += weights[~valid]
    return density, unassigned
