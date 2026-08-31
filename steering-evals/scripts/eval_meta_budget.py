"""eval_meta_budget.py — FAST (<10s): conformal meta-learning of the steering budget.

The reliable steering budget is a QUANTILE of the minimal rank-1 angle y_min
given features: beta(f) s.t. P(beta >= y_min | f) ~ 0.95. Plan:

  fit-targets  (subset of train targets): fit mean regressor mu_hat(x)
  calib-targets (disjoint subset):        residual quantile q = q95(y - mu_hat)
  test-targets (fully held out):          beta = mu_hat(x) + q

Baseline winner to beat: uniform robust recipe 2*alpha*+0.02 (100% reliable,
median excess +0.11 rad above y_min). If mu_hat is accurate (MAE ~0.007) then
q is small and beta is 10x tighter at equal-or-better reliability.

Metrics (held-out targets): rel = P(beta >= y_min), MAE(beta vs y_min),
excess = median(beta - y_min), tight = P(excess < 0.02).

Run: python3 -u eval_meta_budget.py --cache eval_meta_cache_qwen.npz --tag qwen
     python3 -u eval_meta_budget.py --cache eval_meta_cache_gemma.npz --tag gemma
"""
import argparse

import numpy as np
import torch

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='eval_meta_cache_qwen.npz')
    ap.add_argument('--tag', default='qwen')
    ap.add_argument('--test-frac', type=float, default=0.25)
    ap.add_argument('--calib-frac', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    d = np.load(a.cache)
    X, y, meta = d['X'], d['y'], d['meta']
    feats = [str(f) for f in d['feats']]
    T = int(meta[:, 1].max()) + 1
    print(f"== [{a.tag}] conformal meta-budget: {len(y)} pairs, {T} targets ==")

    rng = np.random.default_rng(a.seed)
    all_t = np.arange(T)
    rng.shuffle(all_t)
    n_test = int(T * a.test_frac)
    n_cal = int((T - n_test) * a.calib_frac)
    test_t = set(all_t[:n_test].tolist())
    cal_t = set(all_t[n_test:n_test + n_cal].tolist())
    fit_t = set(all_t[n_test + n_cal:].tolist())
    print(f"  fit {len(fit_t)} targets | calib {len(cal_t)} | test {len(test_t)}")

    def rows(tset):
        m = np.array([m in tset for m in meta[:, 1]])
        return X[m], y[m]

    Xfit, yfit = rows(fit_t)
    Xcal, ycal = rows(cal_t)
    Xte, yte = rows(test_t)

    mu, sd = Xfit.mean(0), Xfit.std(0) + 1e-9
    Zfit, Zcal, Zte = (Xfit - mu) / sd, (Xcal - mu) / sd, (Xte - mu) / sd

    def fit_ridge():
        A = Zfit.T @ Zfit + np.eye(Zfit.shape[1]) * 1.0
        w = np.linalg.solve(A, Zfit.T @ yfit)
        return Zte @ w, Zcal @ w

    def fit_knn():
        from scipy.spatial import cKDTree
        tree = cKDTree(Zfit)
        _, idxs = tree.query(Zcal, k=5)
        mu_cal = yfit[idxs].mean(axis=1)
        _, idxs2 = tree.query(Zte, k=5)
        mu_te = yfit[idxs2].mean(axis=1)
        return mu_te, mu_cal

    def fit_mlp():
        torch.manual_seed(a.seed)
        net = torch.nn.Sequential(
            torch.nn.Linear(Zfit.shape[1], 32), torch.nn.ReLU(),
            torch.nn.Linear(32, 1))
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        Zt, yt = torch.as_tensor(Zfit), torch.as_tensor(yfit, dtype=torch.float32)
        Zc, Ze = torch.as_tensor(Zcal), torch.as_tensor(Zte)
        net.train()
        for _ in range(500):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(Zt).squeeze(1), yt)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            return net(Ze).squeeze(1).numpy(), net(Zc).squeeze(1).numpy()

    print(f"  {'method':>7} {'rel>=0.95':>9} {'MAE':>7} {'excess':>7} {'tight':>6}")
    mlp_beta = None
    for name in ('knn', 'mlp', 'ridge'):
        mu_te, mu_cal = (fit_ridge() if name == 'ridge'
                         else fit_knn() if name == 'knn' else fit_mlp())
        r = ycal - mu_cal
        q = np.quantile(r, 0.95)
        beta = np.clip(mu_te + q, 0.0, 1.2)
        rel = np.mean(beta >= yte - 1e-9)
        mae = np.abs(beta - yte).mean()
        exc = np.median(beta - yte)
        tight = np.mean((beta - yte) < 0.02)
        print(f"  {name + '+q':>7} {rel:>9.3f} {mae:>7.4f} {exc:>7.4f} {tight:>6.3f}")
        if name == 'mlp':
            mlp_beta = beta

    ast = Xte[:, 0]
    bl = {'astar': ast, 'x2': 2 * ast + 0.02,
          'median': np.full(len(yte), np.median(yfit))}
    for name, pr in bl.items():
        pr = np.clip(pr, 0.0, 1.2)
        rel = np.mean(pr >= yte - 1e-9)
        mae = np.abs(pr - yte).mean()
        exc = np.median(pr - yte)
        tight = np.mean((pr - yte) < 0.02)
        print(f"  {name:>7} {rel:>9.3f} {mae:>7.4f} {exc:>7.4f} {tight:>6.3f}")

    res = mlp_beta - np.clip(2 * ast + 0.02, 0.0, 1.2)
    rk = (Xte[:, 8] - mu[8]) / sd[8]
    print(f"  corr(mlp+q - x2, rank_t): {np.corrcoef(res, rk)[0, 1]:+.3f} "
          f"(negative = learner tightens low-blocker targets)")


if __name__ == "__main__":
    main()