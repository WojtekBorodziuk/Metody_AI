"""Wykrywanie anomalnych dni giełdowych - klasyfikacja binarna."""

import sys
import warnings

import numpy as np
from imblearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

from src import data as D
from src import experiment as E
from src import models as M

warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# Parametry eksperymentu
PERIOD = "10y"
SIGMA = 3.0
N_SPLITS = 2
N_REPEATS = 5
ALPHA = 0.05
RANDOM_STATE = 42


def header(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def mean_std_table(scores_metric, clf_names, res_names):
    means = scores_metric.mean(axis=0)
    stds = scores_metric.std(axis=0)
    text = [
        [clf_names[i]] + [f"{means[i, j]:.3f} ({stds[i, j]:.3f})"
                          for j in range(len(res_names))]
        for i in range(len(clf_names))
    ]
    return text, means


# P1 - dane i definicja problemu
def section_p1():
    header("P1: ZBIÓR DANYCH I PROBLEM ROZPOZNAWANIA")
    print(
        "Problem: wykrywanie dni ANOMALNYCH (ekstremalna dzienna stopa zwrotu,\n"
        f"reguła |ret - mean| > {SIGMA}*std). Klasyfikacja binarna niezbalansowana.\n"
        "Cechy są OPÓŹNIONE (z dni t-1 i wcześniej) - brak przecieku informacji."
    )

    datasets = {}
    for ticker, name in D.TICKERS.items():
        df = D.download_prices(ticker, period=PERIOD)
        X, y, dates, prices, feat_names = D.build_dataset(df, sigma=SIGMA)
        n_neg, n_pos, pct = D.class_summary(y)
        datasets[ticker] = dict(
            X=X, y=y, dates=dates, prices=prices, feat_names=feat_names, name=name
        )
        print(
            f"\n{ticker:5s} | {name}"
            f"\n      próbki: {X.shape[0]}, cechy: {X.shape[1]}, "
            f"anomalie: {n_pos} ({pct:.2f}%), normalne: {n_neg}"
        )

    print("\nCechy:", ", ".join(datasets[list(datasets)[0]]["feat_names"]))
    return datasets


# P2/P3 - klasyfikatory i resampling
def section_p2_p3(datasets):
    header("P2/P3: METODY ROZPOZNAWANIA")
    print(
        "Trzy różnorodne klasyfikatory:\n"
        "  k-NN - instancyjny (oparty na odległościach),\n"
        "  RF   - zespołowy (las drzew decyzyjnych),\n"
        "  GNB  - probabilistyczny (Naiwny Klasyfikator Gaussowski).\n"
        "Potok (Pipeline): skalowanie (StandardScaler) -> resampling -> model.\n"
        "Metody resamplingu: None, ROS, RUS, SMOTE."
    )

    from sklearn.naive_bayes import GaussianNB

    demo_ticker = "NVDA"
    d = datasets[demo_ticker]
    X_tr, X_te, y_tr, y_te = train_test_split(
        d["X"], d["y"], test_size=0.2, random_state=RANDOM_STATE, stratify=d["y"],
    )
    print(f"\nDemonstracja (pojedynczy podział, {demo_ticker}, klasyfikator GNB):")
    for label, sampler in [("None ", None), ("SMOTE", M.SMOTE(random_state=RANDOM_STATE))]:
        if sampler is None:
            model = make_pipeline(StandardScaler(), GaussianNB())
        else:
            model = make_pipeline(StandardScaler(), sampler, GaussianNB())
        model.fit(X_tr, y_tr)
        bac = balanced_accuracy_score(y_te, model.predict(X_te))
        print(f"  {label} | balanced accuracy: {bac:.3f}")
    print("  -> efekt resamplingu zależy od danych; w pełnym eksperymencie (CV) SMOTE zwykle pomaga.")


# P4 - walidacja krzyżowa
def section_p4(datasets):
    header(f"P4: EKSPERYMENT (CV {N_REPEATS}x{N_SPLITS}, "
           f"{N_REPEATS * N_SPLITS} foldów)")
    classifiers = M.get_classifiers()
    resamplers = M.get_resamplers()

    results = {}
    for ticker in D.TICKERS:
        print(f"-> Walidacja krzyżowa dla {ticker}...")
        d = datasets[ticker]
        scores, clf_names, res_names = E.run_cv(
            d["X"], d["y"], classifiers, resamplers,
            n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE,
        )
        results[ticker] = dict(scores=scores, clf_names=clf_names,
                               res_names=res_names)
    print("[OK] Eksperyment zakończony.")
    return results


# P5 - tabele wyników
def section_p5(results):
    header("P5: ANALIZA WYNIKÓW")
    for ticker in D.TICKERS:
        r = results[ticker]
        clf_names, res_names = r["clf_names"], r["res_names"]
        print(f"\n----- {ticker} ({D.TICKERS[ticker]}) -----")
        text, means = mean_std_table(r["scores"]["BAC"], clf_names, res_names)
        print(f"\n[BAC] średnia (odch. std.) z {N_REPEATS * N_SPLITS} foldów:")
        print(tabulate(text, headers=[""] + res_names, tablefmt="github"))


# P6 - testy statystyczne i wnioski
def section_p6(results):
    header("P6: ANALIZA STATYSTYCZNA I WNIOSKI")
    print(
        "Porównanie klasyfikatorów przy ustalonym resamplingu = SMOTE,\n"
        "metryka = balanced accuracy. Test normalności (Shapiro-Wilka) oraz\n"
        f"test t-Studenta dla prób zależnych. Poziom istotności alpha={ALPHA}.\n"
    )

    pair_results = {}
    for ticker in D.TICKERS:
        r = results[ticker]
        clf_names = r["clf_names"]
        smote_idx = r["res_names"].index("SMOTE")
        scores_2d = r["scores"]["BAC"][:, :, smote_idx]

        print(f"\n----- {ticker} ({D.TICKERS[ticker]}) -----")

        norm = E.normality_test(scores_2d, clf_names, alpha=ALPHA)
        print("Test normalności Shapiro-Wilka:")
        for n, (stat, p, ok) in norm.items():
            print(f"  {n:5s} | statystyka={stat:.3f} | p-value={p:.3f} "
                  f"| rozkład normalny: {ok}")

        _, _, _, _, means, lines = E.paired_ttests(scores_2d, clf_names, alpha=ALPHA)
        print("\nInterpretacja porównań par (t-Student, prób zależnych):")
        for line in lines:
            print("  - " + line)

        best = clf_names[int(np.argmax(means))]
        print(f"  => Najlepszy klasyfikator na {ticker}: {best} "
              f"(śr. BAC={means.max():.3f}).")

        pair_results[ticker] = E.collect_pair_results(scores_2d, clf_names, alpha=ALPHA)

    # Zbiorcze podsumowanie: w ilu na ile porównań różnica jest istotna
    header("PODSUMOWANIE ISTOTNOŚCI STATYSTYCZNEJ (zbiorczo)")
    summ = E.significance_summary(pair_results, alpha=ALPHA)
    for line in summ["lines"]:
        print(line)
    print(
        "\nWniosek: różnice między klasyfikatorami są istotne statystycznie "
        f"w {summ['n_significant']} na {summ['total']} porównań "
        f"({summ['pct']:.1f}%). Pozostałe różnice mieszczą się w zakresie\n"
        "losowej zmienności walidacji krzyżowej (brak podstaw do odrzucenia H0)."
    )


def main():
    datasets = section_p1()
    section_p2_p3(datasets)
    results = section_p4(datasets)
    section_p5(results)
    section_p6(results)
    header("KONIEC")


if __name__ == "__main__":
    main()
