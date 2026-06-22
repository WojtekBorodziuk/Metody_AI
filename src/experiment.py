"""Walidacja krzyżowa i testy statystyczne (scipy)."""

import numpy as np
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import make_pipeline


METRICS = ["ACC", "BAC"]


# Walidacja krzyżowa dla każdej pary klasyfikator x resampling
def run_cv(X, y, classifiers: dict, resamplers: dict,
           n_splits: int = 2, n_repeats: int = 5, random_state: int = 42):
    clf_names = list(classifiers.keys())
    res_names = list(resamplers.keys())
    n_clf = len(clf_names)
    n_res = len(res_names)
    n_folds = n_splits * n_repeats

    scores = {m: np.zeros((n_folds, n_clf, n_res)) for m in METRICS}
    rkf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )

    for c_idx, (cn, clf) in enumerate(classifiers.items()):
        for r_idx, (rn, sampler) in enumerate(resamplers.items()):
            if sampler is not None:
                model = make_pipeline(StandardScaler(), sampler, clf)
            else:
                model = make_pipeline(StandardScaler(), clf)

            try:
                cv_res = cross_validate(
                    model, X, y, cv=rkf,
                    scoring={"ACC": "accuracy", "BAC": "balanced_accuracy"},
                    n_jobs=-1,
                )
                scores["ACC"][:, c_idx, r_idx] = cv_res["test_ACC"]
                scores["BAC"][:, c_idx, r_idx] = cv_res["test_BAC"]
            except Exception as exc:
                print(f"    [WARN] clf={cn}, res={rn}: {exc}")
                scores["ACC"][:, c_idx, r_idx] = np.nan
                scores["BAC"][:, c_idx, r_idx] = np.nan

    return scores, clf_names, res_names


# Test normalności rozkładu wyników w foldach (Shapiro-Wilka)
def normality_test(scores_2d, clf_names, alpha: float = 0.05) -> dict:
    result = {}
    for i, name in enumerate(clf_names):
        col = scores_2d[:, i]
        col = col[~np.isnan(col)]
        if len(col) < 3:
            result[name] = (np.nan, np.nan, "N/A")
            continue
        stat, p = stats.shapiro(col)
        result[name] = (float(stat), float(p), "TAK" if p > alpha else "NIE")
    return result


# Porównania par klasyfikatorów - test t-Studenta dla prób zależnych
def paired_ttests(scores_2d, clf_names, alpha: float = 0.05):
    n_clf = len(clf_names)
    t_stat = np.zeros((n_clf, n_clf))
    p_val = np.ones((n_clf, n_clf))
    better = np.zeros((n_clf, n_clf), dtype=bool)
    sig = np.zeros((n_clf, n_clf), dtype=bool)
    means = np.nanmean(scores_2d, axis=0)
    lines = []

    for i in range(n_clf):
        for j in range(i + 1, n_clf):
            a = scores_2d[:, i]
            b = scores_2d[:, j]
            mask = ~(np.isnan(a) | np.isnan(b))
            if mask.sum() < 3:
                continue
            t, p = stats.ttest_rel(a[mask], b[mask])
            t_stat[i, j] = t
            t_stat[j, i] = -t
            p_val[i, j] = p
            p_val[j, i] = p
            sig[i, j] = sig[j, i] = p < alpha
            better[i, j] = means[i] > means[j]
            better[j, i] = means[j] > means[i]

            if p < alpha:
                winner = clf_names[i] if means[i] > means[j] else clf_names[j]
                lines.append(
                    f"{clf_names[i]} vs {clf_names[j]}: "
                    f"t={t:.3f}, p={p:.4f} (*) => {winner} istotnie lepszy (α={alpha})"
                )
            else:
                lines.append(
                    f"{clf_names[i]} vs {clf_names[j]}: "
                    f"t={t:.3f}, p={p:.4f} (brak istotnej różnicy, α={alpha})"
                )

    return t_stat, p_val, better, sig, means, lines


# Wynik porównań par dla jednej spółki (na potrzeby agregacji)
def collect_pair_results(scores_2d, clf_names, alpha: float = 0.05):
    _, p_val, _, _, means, _ = paired_ttests(scores_2d, clf_names, alpha=alpha)
    n_clf = len(clf_names)
    out = []
    for i in range(n_clf):
        for j in range(i + 1, n_clf):
            p = float(p_val[i, j])
            sig = p < alpha
            winner = clf_names[i] if means[i] >= means[j] else clf_names[j]
            out.append({
                "pair": f"{clf_names[i]} vs {clf_names[j]}",
                "p": p,
                "significant": bool(sig),
                "winner": winner if sig else None,
                "diff": float(abs(means[i] - means[j])),
            })
    return out


# Zbiorcze zliczenie: w ilu na ile porównań różnica jest istotna
def significance_summary(all_results, alpha: float = 0.05):
    total = 0
    n_significant = 0
    per_pair = {}
    per_ticker = {}

    for ticker, entries in all_results.items():
        per_ticker.setdefault(ticker, [0, 0])
        for e in entries:
            total += 1
            per_ticker[ticker][1] += 1
            pp = per_pair.setdefault(e["pair"], [0, 0])
            pp[1] += 1
            if e["significant"]:
                n_significant += 1
                per_ticker[ticker][0] += 1
                pp[0] += 1

    pct = 100.0 * n_significant / total if total else 0.0

    lines = [
        f"Łącznie istotnych statystycznie: {n_significant}/{total} "
        f"porównań ({pct:.1f}%), przy α={alpha}.",
        "Rozkład istotności wg pary klasyfikatorów:",
    ]
    for pair, (ns, nt) in per_pair.items():
        lines.append(f"  - {pair}: istotne w {ns}/{nt} spółkach")
    lines.append("Rozkład istotności wg spółki:")
    for ticker, (ns, nt) in per_ticker.items():
        lines.append(f"  - {ticker}: istotne {ns}/{nt} porównań")

    return {
        "total": total,
        "n_significant": n_significant,
        "pct": pct,
        "per_pair": per_pair,
        "per_ticker": per_ticker,
        "lines": lines,
    }
