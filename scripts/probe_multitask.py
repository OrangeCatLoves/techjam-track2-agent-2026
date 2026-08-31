"""Do auxiliary heads on the shared embeddings help the main task?

The features experiment showed the model OVERFITS once given denser signal --
train/valid gap went 0.0065 -> 0.0255 with validation flat. So more capacity is
the wrong medicine. Auxiliary heads are the opposite: extra tasks CONSTRAIN the
shared embedding table rather than widening it, which is regularisation.

Architecture. One shared embedding table V, and per task a linear head plus a
scalar gain on the interaction term:

    E = V[X]                       (B,F,k)
    S = E.sum(1)                   (B,k)
    inter = 0.5*((S**2).sum(1) - (E**2).sum((1,2)))
    z_t = b_t + W_t[X].sum(1) + a_t * inter

V is shared, so every task's gradient flows into it; W_t, b_t and a_t are private.
Only the main head is ever scored.

    loss = logloss(long_view) + lam * mean_t logloss(aux_t)

lam = 0 makes the auxiliary heads inert, so it must reproduce the reference FM --
that is the check that this reimplementation is faithful before any conclusion is
drawn from it.

is_click / is_like / is_profile_enter are outcomes of the impression. CLAUDE.md
section 7.2 permits them as auxiliary TARGETS and forbids them as inputs; decision
D22 records the same reading for play_time_ms. Train rows only; nothing here
reaches a feature vector, and valid/test outcomes are never read.
"""
import csv, time
import numpy as np
from harness import data as hdata
from harness import evaluate as hevaluate
from harness.models import runners as R

D = str(hdata.data_dir())
AUX = ['is_click', 'is_like', 'is_profile_enter']
SEED, BATCH, K, LR, L2 = 0, 8192, 16, 0.001, 1e-6

def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def load_aux(splits):
    """Auxiliary targets for TRAIN rows, aligned and verified row by row."""
    got = {c: [] for c in AUX}
    uid, vid, lv = [], [], []
    for fn in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(f'{D}/{fn}', newline='') as fh:
            for r in csv.DictReader(fh):
                if 20220408 <= int(r['date']) <= 20220421:
                    for c in AUX: got[c].append(1.0 if r[c] != '0' else 0.0)
                    uid.append(r['user_id']); vid.append(r['video_id'])
                    lv.append(1 if r['long_view'] != '0' else 0)
    rows = splits['train']
    if len(rows) != len(uid): raise RuntimeError('row count mismatch')
    for name, a, b in (('user_id', [r[hdata.IDX_USER] for r in rows], uid),
                       ('video_id', [r[hdata.IDX_VIDEO] for r in rows], vid),
                       ('long_view', [r[hdata.IDX_LABEL] for r in rows], lv)):
        if a != b: raise RuntimeError(f'{name} misaligned')
    return np.array([got[c] for c in AUX], dtype=np.float32)   # (T, N)

class MultiTaskFM:
    def __init__(self, dim, n_tasks, k=K, lr=LR, l2=L2, seed=SEED):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros((n_tasks, dim), dtype=np.float32)
        self.b = np.zeros(n_tasks, dtype=np.float32)
        self.a = np.ones(n_tasks, dtype=np.float32)
        self.lr, self.l2, self.T = lr, l2, n_tasks
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0

    def _inter(self, X):
        E = self.V[X]; S = E.sum(1)
        return 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2))), E, S

    def predict(self, X, bs=200_000):
        out = []
        for i in range(0, len(X), bs):
            xb = X[i:i + bs]
            inter, _, _ = self._inter(xb)
            out.append(self.b[0] + self.W[0][xb].sum(1) + self.a[0] * inter)
        return np.concatenate(out)

    def step(self, X, Y, lam):
        """Y is (T, B): row 0 the main task, the rest auxiliary."""
        B = Y.shape[1]
        inter, E, S = self._inter(X)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        g_inter = np.zeros(B, dtype=np.float64)
        total = 0.0
        for t in range(self.T):
            z = self.b[t] + self.W[t][X].sum(1) + self.a[t] * inter
            p = sigmoid(z)
            w = 1.0 if t == 0 else lam / max(1, self.T - 1)
            g = w * (p - Y[t]) / B
            np.add.at(gW[t], X, g[:, None])
            self.b[t] -= self.lr * g.sum()
            self.a[t] -= self.lr * float((g * inter).sum())
            g_inter += g * self.a[t]
            total += w * float(-np.mean(Y[t] * np.log(p + 1e-9)
                                        + (1 - Y[t]) * np.log(1 - p + 1e-9)))
        np.add.at(gV, X, g_inter[:, None, None].astype(np.float32) * (S[:, None, :] - E))
        gV += self.l2 * self.V; gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        return total

splits = hdata.load()
enc, dim = hdata.encode(splits)
Xtr, ytr, _ = enc['train']; Xva, yva, uva = enc['valid']
aux = load_aux(splits)
print(f'aux targets aligned on all {aux.shape[1]:,} train rows: {AUX}')
Y = np.vstack([ytr[None, :].astype(np.float32), aux])

t0 = time.time()
print()
for lam in (0.0, 0.1, 0.3, 1.0, 3.0):
    model = MultiTaskFM(dim, n_tasks=Y.shape[0], seed=SEED)
    rng = np.random.default_rng(SEED)
    best, bad, best_ep = -1.0, 0, 0
    for epoch in range(1, 41):
        order = rng.permutation(Y.shape[1])
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            model.step(Xtr[idx], Y[:, idx], lam)
        s = float(hevaluate.evaluate(uva, yva, model.predict(Xva))['primary'])
        if s > best + 1e-5: best, bad, best_ep = s, 0, epoch
        else:
            bad += 1
            if bad >= 4: break
    tag = 'lam 0.0 (control, aux inert)' if lam == 0 else f'lam {lam}'
    print(f'  {tag:<32} {best:.4f}  (best epoch {best_ep})', flush=True)
print()
print(f'elapsed {time.time()-t0:.0f}s')
