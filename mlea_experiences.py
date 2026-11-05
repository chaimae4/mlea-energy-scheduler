import time
import heapq
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor


P_IDLE = 100.0
P_MAX  = 250.0
ALPHA  = 1.0          # M1 = 1.0  |  M3 = 1.4
E_BOOT = 500.0
IDLE_TIMEOUT = 600.0

N_SEEDS   = 10
M_VMS     = 50
WORKLOADS = [300, 800, 1500]
CPU_CHOICES = [4, 8, 16, 32]
MEM_CHOICES = [8, 16, 32, 64]
FICHIER_TACHES = "taches_gct.csv"

FEATURES = ["task_cpu", "task_mem", "duration", "sched_class",
            "priority", "vm_cpu_cap", "vm_mem_cap", "u_before"]


def puissance(u):
    return P_IDLE + (P_MAX - P_IDLE) * np.power(np.clip(u, 0, 1), ALPHA)


def energie_incrementale(u_before, u_after, duration_s, vm_off):
    dE = (puissance(u_after) - puissance(u_before)) * duration_s
    return dE + np.where(vm_off, E_BOOT, 0.0)



def charger_taches(chemin=FICHIER_TACHES):
    df = pd.read_csv(chemin)
    df = df.dropna(subset=["cpu", "mem", "duration_s"])
    df = df[(df["cpu"] > 0) & (df["duration_s"] > 0)]
    df = df.sort_values("t_start").reset_index(drop=True)
    print(f"{len(df)} taches chargees")
    return df



def construire_dataset(taches, rng):
    n = len(taches)
    vm_cpu_cap = rng.choice(CPU_CHOICES, n).astype(float)
    vm_mem_cap = rng.choice(MEM_CHOICES, n).astype(float)
    u_before   = rng.uniform(0.0, 0.9, n)
    vm_off     = rng.random(n) < 0.05
    u_after = np.clip(u_before + taches["cpu"].values / vm_cpu_cap, 0, 1)
    X = pd.DataFrame({
        "task_cpu":    taches["cpu"].values,
        "task_mem":    taches["mem"].values,
        "duration":    np.log1p(taches["duration_s"].values),
        "sched_class": taches["sched_class"].values,
        "priority":    taches["priority"].values,
        "vm_cpu_cap":  vm_cpu_cap,
        "vm_mem_cap":  vm_mem_cap,
        "u_before":    u_before,
    })
    dE = energie_incrementale(u_before, u_after,
                              taches["duration_s"].values, vm_off)
    y = np.log1p(np.maximum(dE, 0.0))
    return X, y


def entrainer(X, y):
    y = np.asarray(y)
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    r2_folds = []
    for tr, te in cv.split(X):
        m = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                         reg_lambda=1.0, n_jobs=-1, random_state=0)
        m.fit(X.iloc[tr], y[tr])
        r2_folds.append(r2_score(y[te], m.predict(X.iloc[te])))
    r2_folds = np.array(r2_folds)
    print(f"5-fold CV : R2 (log) = {r2_folds.mean():.4f} +/- {r2_folds.std():.4f}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    m2 = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                      reg_lambda=1.0, n_jobs=-1, random_state=0).fit(Xtr, ytr)
    mae_ws = np.mean(np.abs(np.expm1(m2.predict(Xte)) - np.expm1(yte)))
    print(f"MAE (echelle reelle) = {mae_ws:.2f} W.s")

    model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                         reg_lambda=1.0, n_jobs=-1, random_state=0)
    model.fit(X, y)
    return model


def nouvelle_flotte(rng):
    return {
        "cpu_cap":  rng.choice(CPU_CHOICES, M_VMS).astype(float),
        "mem_cap":  rng.choice(MEM_CHOICES, M_VMS).astype(float),
        "cpu_used": np.zeros(M_VMS),
        "mem_used": np.zeros(M_VMS),
        "last_end": np.zeros(M_VMS),
        "off":      np.zeros(M_VMS, bool),
    }


def simuler(taches, mode, model=None, rng=None, duree_decision=None):
    col_decision = duree_decision if duree_decision is not None else "duration_s"
    flotte = nouvelle_flotte(rng)
    rr_pointer, energie_totale = 0, 0.0
    latences = []
    en_cours = []

    for _, t in taches.iterrows():
        t_now = t["t_start"] / 1e6

        while en_cours and en_cours[0][0] <= t_now:
            fin, j, c, m = heapq.heappop(en_cours)
            flotte["cpu_used"][j] -= c
            flotte["mem_used"][j] -= m
            flotte["last_end"][j] = fin

        inactives = (t_now - flotte["last_end"] > IDLE_TIMEOUT) & \
                    (flotte["cpu_used"] == 0)
        flotte["off"] = inactives

        faisable = ((flotte["cpu_used"] + t["cpu"] <= flotte["cpu_cap"]) &
                    (flotte["mem_used"] + t["mem"] <= flotte["mem_cap"]))
        idx = np.where(faisable)[0]
        if len(idx) == 0:
            continue

        if mode == "RR":
            j = idx[rr_pointer % len(idx)]
            rr_pointer += 1
        elif mode == "LC":
            j = idx[np.argmin(flotte["cpu_used"][idx] / flotte["cpu_cap"][idx])]
        elif mode == "MBFD":
            reste = flotte["cpu_cap"][idx] - flotte["cpu_used"][idx] - t["cpu"]
            j = idx[np.argmin(reste)]
        else:  # MLEA
            t0 = time.perf_counter()
            Xq = pd.DataFrame({
                "task_cpu":    np.full(len(idx), t["cpu"]),
                "task_mem":    np.full(len(idx), t["mem"]),
                "duration":    np.full(len(idx), np.log1p(t[col_decision])),
                "sched_class": np.full(len(idx), t["sched_class"]),
                "priority":    np.full(len(idx), t["priority"]),
                "vm_cpu_cap":  flotte["cpu_cap"][idx],
                "vm_mem_cap":  flotte["mem_cap"][idx],
                "u_before":    flotte["cpu_used"][idx] / flotte["cpu_cap"][idx],
            })
            preds = model.predict(Xq[FEATURES].values)
            latences.append((time.perf_counter() - t0) * 1000)
            j = idx[np.argmin(preds)]

        u_b = flotte["cpu_used"][j] / flotte["cpu_cap"][j]
        u_a = (flotte["cpu_used"][j] + t["cpu"]) / flotte["cpu_cap"][j]
        energie_totale += float(energie_incrementale(
            u_b, u_a, t["duration_s"], flotte["off"][j]))

        flotte["cpu_used"][j] += t["cpu"]
        flotte["mem_used"][j] += t["mem"]
        flotte["last_end"][j] = t_now + t["duration_s"]
        flotte["off"][j] = False
        heapq.heappush(en_cours, (flotte["last_end"][j], j, t["cpu"], t["mem"]))

    lat_moy = float(np.mean(latences)) if latences else 0.0
    return energie_totale, lat_moy



def main():
    taches = charger_taches()

    rng = np.random.default_rng(42)
    X, y = construire_dataset(taches, rng)
    model = entrainer(X, y)

    resultats = {m: {N: [] for N in WORKLOADS}
                 for m in ["RR", "LC", "MBFD", "MLEA"]}

    for N in WORKLOADS:
        latence = 0.0
        for seed in range(N_SEEDS):
            r = np.random.default_rng(seed)
            ech = taches.sample(n=N, random_state=seed).sort_values("t_start")
            for mode in ["RR", "LC", "MBFD", "MLEA"]:
                E, lat = simuler(ech, mode, model if mode == "MLEA" else None, r)
                resultats[mode][N].append(E)
                if mode == "MLEA":
                    latence = lat

        E = {m: np.mean(resultats[m][N]) for m in resultats}
        print(f"\nN = {N} (moyenne sur {N_SEEDS} graines)")
        for m in ["RR", "LC", "MBFD", "MLEA"]:
            print(f"  {m:>5} : {E[m]:>14,.0f} W.s   "
                  f"(vs RR : {100*(E['RR']-E[m])/E['RR']:5.1f} %)")
        for m in ["RR", "LC", "MBFD"]:
            stat, p = wilcoxon(resultats["MLEA"][N], resultats[m][N])
            print(f"  Wilcoxon MLEA vs {m:>5} : p = {p:.2e}")
        print(f"  Latence d'inference moyenne : {latence:.2f} ms/tache")


    print("\n=== Ablation : duree oracle vs predite ===")
    med = taches.groupby("sched_class")["duration_s"].median()

    for N in WORKLOADS:
        E_oracle_runs, E_pred_runs = [], []
        for seed in range(N_SEEDS):
            r = np.random.default_rng(seed)
            ech = taches.sample(n=N, random_state=seed).sort_values("t_start")

            E_o, _ = simuler(ech, "MLEA", model, r)  # oracle

            ech_p = ech.copy()
            ech_p["duration_decision"] = ech_p["sched_class"].map(med)
            E_p, _ = simuler(ech_p, "MLEA", model, r,
                             duree_decision="duration_decision")  # predit

            E_oracle_runs.append(E_o)
            E_pred_runs.append(E_p)
        E_o, E_p = np.mean(E_oracle_runs), np.mean(E_pred_runs)
        print(f"  N={N}: oracle={E_o:,.0f} | predit={E_p:,.0f} "
              f"| ecart = {100*(E_p-E_o)/E_o:+.1f} %")


main()