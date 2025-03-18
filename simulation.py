import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


def generate_gbm_paths(N, T, s0, mean, volatility, h=1, plot=False):
    """
    Generuje ścieżki ceny przy użyciu geometrycznego ruchu Browna.

    Parameters:
    -----------
    N : int
        Liczba ścieżek.
    T : int
        Horyzont czasowy (liczba dni).
    s0 : float
        Początkowa cena aktywa.
    mean : float
        Średnia logarytmicznych zwrotów dziennych.
    volatility : float
        Odchylenie standardowe logarytmicznych zwrotów dziennych.
    h : float, optional
        Krok czasowy. Domyślnie 1 dzień.
    plot : bool, optional
        Czy wykreślić ścieżki. Domyślnie False.

    Returns:
    --------
    paths : ndarray
        Macierz ścieżek cenowych (N x (T/h + 1)).
    time : ndarray
        Macierz punktów czasowych.
    """
    dt = h
    n = int(T / h)
    mu = mean * h
    sigma = volatility * np.sqrt(h)

    paths = np.exp((mu - sigma ** 2 / 2) * dt +
                   sigma * np.random.normal(0, np.sqrt(dt), size=(N, n)))
    paths = np.hstack([np.ones((N, 1)), paths])
    paths = s0 * paths.cumprod(axis=1)

    time = np.linspace(0, T, n + 1)
    matrixtime = np.full(shape=(N, n + 1), fill_value=time)

    if plot is True:
        plt.figure(figsize=(12, 8))
        plt.plot(matrixtime.T, paths.T, alpha=0.3)
        plt.title(f'Symulowane ścieżki ({N}) - Geometryczny ruch Browna')
        plt.xlabel('Czas (dni)')
        plt.ylabel('Wartość WIG20')
        plt.show()

    return paths, matrixtime


def bootstrap_paths(N, T, s0, historical_data, plot=False):
    """
    Generuje ścieżki cenowe metodą bootstrapu z historycznych zwrotów.

    Parameters:
    -----------
    N : int
        Liczba ścieżek.
    T : int
        Horyzont czasowy (liczba dni).
    s0 : float
        Początkowa cena aktywa.
    historical_data : DataFrame
        Dane historyczne z cenami.
    plot : bool, optional
        Czy wykreślić ścieżki. Domyślnie False.

    Returns:
    --------
    paths : ndarray
        Macierz ścieżek cenowych.
    time : ndarray
        Macierz punktów czasowych.
    """
    sample = np.log(historical_data.Close / historical_data.Close.shift(1)).dropna()  # log returns
    rng = np.random.default_rng()  # Inicjalizacja generatora
    rates = np.exp(rng.choice(sample, size=(N, T), replace=True, shuffle=False))

    # Kumulowanie zwrotów
    paths = s0 * np.cumprod(rates, axis=1)
    paths = np.hstack([np.ones((N, 1)) * s0, paths])  # Dodaj początkową cenę

    time = np.arange(0, T + 1)
    matrixtime = np.full(shape=(N, T + 1), fill_value=time)

    if plot is True:
        plt.figure(figsize=(12, 8))
        plt.plot(matrixtime.T, paths.T, alpha=0.3)
        plt.title(f'Symulowane ścieżki ({N}) - Bootstrap')
        plt.xlabel('Czas (dni)')
        plt.ylabel('Wartość WIG20')
        plt.show()

    return paths, matrixtime


def generate_correlated_gbm_paths(N, T, s0_vec, mean_vec, vol_vec, corr_matrix, h=1):
    """
    Generuje ścieżki cen dla dwóch lub więcej aktywów przy użyciu skorelowanego geometrycznego ruchu Browna.

    Parameters:
    -----------
    N : int
        Liczba ścieżek.
    T : int
        Horyzont czasowy (liczba dni).
    s0_vec : list or ndarray
        Wektor początkowych cen aktywów.
    mean_vec : list or ndarray
        Wektor średnich logarytmicznych zwrotów dziennych.
    vol_vec : list or ndarray
        Wektor zmienności (odchylenie standardowe logarytmicznych zwrotów dziennych).
    corr_matrix : ndarray
        Macierz korelacji między aktywami.
    h : float, optional
        Krok czasowy. Domyślnie 1 dzień.

    Returns:
    --------
    paths : dict
        Słownik zawierający macierze ścieżek cenowych dla każdego aktywa.
    time : ndarray
        Macierz punktów czasowych.
    """
    dt = h
    n = int(T / h)
    num_assets = len(s0_vec)

    # Przekształcenie wektorów na tablice numpy
    s0_vec = np.array(s0_vec)
    mean_vec = np.array(mean_vec)
    vol_vec = np.array(vol_vec)

    # Parametry dla procesu GBM
    mu = mean_vec * h
    sigma = vol_vec * np.sqrt(h)

    # Dekompozycja Choleskiego macierzy korelacji
    L = np.linalg.cholesky(corr_matrix)

    # Inicjalizacja ścieżek
    paths = {}
    for i in range(num_assets):
        paths[f'asset_{i}'] = np.zeros((N, n + 1))
        paths[f'asset_{i}'][:, 0] = s0_vec[i]

    # Generowanie ścieżek
    for t in range(1, n + 1):
        # Generowanie nieskorelowanych losowych liczb
        Z = np.random.normal(0, 1, size=(N, num_assets))

        # Korelacja liczb losowych
        correlated_Z = Z @ L.T

        # Aktualizacja cen dla każdego aktywa
        for i in range(num_assets):
            paths[f'asset_{i}'][:, t] = paths[f'asset_{i}'][:, t - 1] * np.exp(
                (mu[i] - 0.5 * sigma[i] ** 2) * dt + sigma[i] * correlated_Z[:, i]
            )

    # Przygotowanie macierzy czasu
    time = np.linspace(0, T, n + 1)



    return paths, time

def wallet(paths, sigma, r, E, type='call', n_hedge=252):
    """
    Generuje portfel hedgujący dla opcji i oblicza zysk/stratę dla każdej ścieżki cenowej.

    Parameters:
    -----------
    paths: ndarray
        Macierz ścieżek ceny aktywa kształtu (N, T), gdzie N to liczba ścieżek, a T to liczba dni
    sigma: float
        Wartość zmienności (volatility)
    r: float
        Stopa procentowa wolna od ryzyka
    E: float
        Cena wykonania opcji (strike)
    type: str
        Typ opcji: 'call' lub 'put'
    n_hedge: int
        Liczba rehedge'ingów w całym okresie

    Returns:
    --------
    pnl: ndarray
        Macierz zysków/strat dla każdej ze ścieżek na koniec każdego dnia, rozmiaru identycznego jak paths
    time_grid: ndarray
        Wektor czasu odpowiadający punktom rebalansowania portfela
    """
    N, T = paths.shape  # N - liczba ścieżek, T - liczba dni
    dt = 1 / 252  # zakładamy rok handlowy (252 dni)
    tau = T * dt  # całkowity czas w latach

    # Ustalenie punktów rehedgingu
    if n_hedge == T:
        # Rehedging dokładnie co jeden dzień
        hedge_indices = np.arange(T)
        intervals_per_day = 1
    elif n_hedge < T:
        # Rehedging rzadziej niż codziennie
        if T % n_hedge != 0:
            # Znajdujemy najbliższą mniejszą wartość n_hedge, która dzieli T
            while T % n_hedge != 0:
                n_hedge -= 1
            print(f"Dostosowano n_hedge do {n_hedge}, aby było dzielnikiem T={T}")

        # Indeksy dni, w których wykonujemy rehedging
        step = T // n_hedge
        hedge_indices = np.arange(0, T, step)
        intervals_per_day = 1 / step
    else:
        # Rehedging częściej niż codziennie (wielokrotność T)
        intervals_per_day = n_hedge // T
        if n_hedge % T != 0:
            n_hedge = intervals_per_day * T  # Upewniamy się, że n_hedge jest wielokrotnością T
            print(f"Dostosowano n_hedge do {n_hedge}, aby było wielokrotnością T={T}")

        # Tworzymy rozszerzoną siatkę dla rehedgingu wewnątrz dnia
        hedge_indices = np.arange(T)

    # Inicjalizacja macierzy wyników
    pnl = np.zeros_like(paths)
    time_grid = np.arange(T) * dt

    # Delta dla pierwszego dnia (t=0)
    time_to_maturity = tau - 0
    S0 = paths[:, 0]
    d1_0 = (np.log(S0 / E) + (r + 0.5 * sigma ** 2) * time_to_maturity) / (sigma * np.sqrt(time_to_maturity))

    if type == 'call':
        delta_0 = norm.cdf(d1_0)
    elif type == 'put':
        delta_0 = norm.cdf(d1_0) - 1
    else:
        raise ValueError("Typ opcji musi być 'call' lub 'put'")

    # Inicjalizacja portfela
    delta_prev = delta_0
    stock_position = delta_prev  # Pozycja w akcjach (delta)
    cash_position = -delta_prev * S0  # Pozycja gotówkowa (założenie: samofinansujący się portfel)

    for t in range(1, T):
        # Obecna wartość akcji
        S_t = paths[:, t]

        # Aktualizacja wartości gotówki (uwzględniamy oprocentowanie)
        cash_position = cash_position * np.exp(r * dt)

        # Obliczamy wartość portfela przed rebalansowaniem
        portfolio_value_before = stock_position * S_t + cash_position

        # Sprawdzamy, czy ten dzień jest dniem rehedgingu
        if t in hedge_indices:
            # Obliczamy nową deltę
            time_to_maturity = tau - (t * dt)
            d1_t = (np.log(S_t / E) + (r + 0.5 * sigma ** 2) * time_to_maturity) / (sigma * np.sqrt(time_to_maturity))

            if type == 'call':
                delta_t = norm.cdf(d1_t)
            elif type == 'put':
                delta_t = norm.cdf(d1_t) - 1

            # Jeśli mamy więcej niż jeden rehedging na dzień
            if intervals_per_day > 1:
                # Wykonujemy rehedging wielokrotnie w ciągu dnia
                for i in range(intervals_per_day):
                    # Możemy założyć liniową interpolację ceny w ciągu dnia
                    # lub po prostu wielokrotne użycie tej samej ceny
                    # Tu używamy tej samej ceny dla uproszczenia

                    # Obliczamy zmianę w pozycji akcji
                    delta_change = delta_t - delta_prev

                    # Aktualizacja pozycji w akcjach
                    stock_position = delta_t

                    # Aktualizacja pozycji gotówkowej (zakładamy samofinansujący się portfel)
                    cash_position -= delta_change * S_t

                    # Aktualizacja poprzedniej delty
                    delta_prev = delta_t
            else:
                # Standardowy rehedging raz dziennie lub rzadziej
                # Obliczamy zmianę w pozycji akcji
                delta_change = delta_t - delta_prev

                # Aktualizacja pozycji w akcjach
                stock_position = delta_t

                # Aktualizacja pozycji gotówkowej (zakładamy samofinansujący się portfel)
                cash_position -= delta_change * S_t

                # Aktualizacja poprzedniej delty
                delta_prev = delta_t

        # Obliczamy wartość portfela po rebalansowaniu
        portfolio_value_after = stock_position * S_t + cash_position

        # Zapisujemy zysk/stratę dla danego dnia
        pnl[:, t] = portfolio_value_after - portfolio_value_before

    return pnl, time_grid

