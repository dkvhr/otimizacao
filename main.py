import numpy as np
import time
import sys

class SimplexSolver:
    def __init__(self, c, A, b, senses, optimize='max', verbose=False):
        self.c = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.senses = senses
        self.optimize = optimize.lower()
        self.verbose = verbose
        if self.optimize not in ('max', 'min'):
            raise ValueError("otimizacao deve ser 'max' ou 'min'.")
        if self.optimize == 'min':  # convert min to max
            self.c = -self.c
        self._build_tableau()

    def _build_tableau(self):
        m, n = self.A.shape
        slack_count = sum(s in ('<=', '>=') for s in self.senses)
        art_count = sum(s in ('>=', '=') for s in self.senses)
        total_cols = n + slack_count + art_count + 1
        T = np.zeros((m+1, total_cols))
        self.slack_idx, self.art_idx = [], []
        sc = ac = 0
        for i in range(m):
            T[i, :n] = self.A[i]
            T[i, -1] = self.b[i]
            if self.senses[i] == '<=':
                T[i, n+sc] = 1; self.slack_idx.append(n+sc); sc += 1
            elif self.senses[i] == '>=':
                T[i, n+sc] = -1; self.slack_idx.append(n+sc); sc += 1
                T[i, n+slack_count+ac] = 1; self.art_idx.append(n+slack_count+ac); ac += 1
            else:  # '='
                T[i, n+slack_count+ac] = 1; self.art_idx.append(n+slack_count+ac); ac += 1
        T[m, :n] = -self.c
        self.tableau = T
        self.basis = [None]*m
        sc = ac = 0
        for i, s in enumerate(self.senses):
            if s == '<=':
                self.basis[i] = n+sc; sc += 1
            elif s == '>=':
                self.basis[i] = n+slack_count+ac; ac += 1
            else:
                self.basis[i] = n+slack_count+ac; ac += 1
        self.phase_one_needed = bool(self.art_idx)
        if self.verbose:
            print("tabela inicial:")
            print(self.tableau)

    def solve(self):
        start = time.time()
        self.iterations = 0
        if self.phase_one_needed:
            if self.verbose: print("\n[+] fase I")
            self._phase_one()
        if self.verbose: print("\n[+] fase II")
        status = self._optimize()
        elapsed = time.time() - start
        obj = self.tableau[-1, -1]
        if self.optimize == 'min': obj = -obj
        return {'status': status, 'objective': obj,
                'iterations': self.iterations, 'time': elapsed}

    def _phase_one(self):
        m, ncol = self.tableau.shape
        phase_obj = np.zeros(ncol)
        for i, col in enumerate(self.basis):
            if col in self.art_idx:
                phase_obj -= self.tableau[i]
        self.tableau[-1] = phase_obj
        if self.verbose:
            print("objetivo fase I:", phase_obj)
        self._pivot_loop(phase=1)
        if abs(self.tableau[-1, -1]) > 1e-6:
            raise ValueError("problema inviavel (fase 1 nao zerou o custo)")
        # removendo artificiais
        keep = [j for j in range(self.tableau.shape[1]) if j not in self.art_idx]
        self.tableau = self.tableau[:, keep]
        # ajustando base
        new_basis = []
        for b in self.basis:
            new_basis.append(None if b in self.art_idx else keep.index(b))
        self.basis = new_basis
        # restaurando objetivo original
        m, ncol = self.tableau.shape
        obj_row = np.zeros(ncol); obj_row[:len(self.c)] = -self.c
        self.tableau[-1] = obj_row
        for i, bcol in enumerate(self.basis):
            if bcol is not None:
                coef = self.tableau[-1, bcol]
                if abs(coef) > 1e-8:
                    self.tableau[-1] -= coef * self.tableau[i]
        if self.verbose:
            print("tabela apos a fase I (sem artificiais):")
            print(self.tableau)

    def _optimize(self):
        return self._pivot_loop(phase=2)

    def _pivot_loop(self, phase):
        m, ncol = self.tableau.shape
        while True:
            cost = self.tableau[-1, :-1]
            idx = np.where(cost < -1e-8)[0]
            if len(idx) == 0:
                if self.verbose: print("[+] solucao otima encontrada!!")
                return 'optimal'
            ent = idx[np.argmin(cost[idx])]
            col = self.tableau[:-1, ent]
            rhs = self.tableau[:-1, -1]
            valid = [(rhs[i]/col[i], i) for i in range(m-1) if col[i] > 1e-8]
            if not valid:
                if self.verbose: print("problema ilimitado")
                return 'unbounded'
            ratio, pivot_row = min(valid)
            if self.verbose:
                self.iterations += 1
                print(f"\n[+] iteracao {self.iterations} (Fase {phase}):")
                print(f"variavel entra: coluna {ent}, sai: linha {pivot_row}")
                print(self.tableau)
            self._pivot(pivot_row, ent)
            if self.verbose:
                print("tabela apos pivo:")
                print(self.tableau)
        
    def _pivot(self, row, col):
        m, ncol = self.tableau.shape
        pivot_val = self.tableau[row, col]
        self.tableau[row] /= pivot_val
        for i in range(m):
            if i != row:
                factor = self.tableau[i, col]
                self.tableau[i] -= factor * self.tableau[row]
        self.basis[row] = col

# leitura de entrada padrão
def main():
    data = sys.stdin.read().strip().split()
    it = iter(data)
    optimize = next(it)
    n, m = int(next(it)), int(next(it))
    c = [float(next(it)) for _ in range(n)]
    A, senses, b = [], [], []
    for _ in range(m):
        row = [float(next(it)) for _ in range(n)]
        s = next(it); val = float(next(it))
        A.append(row); senses.append(s); b.append(val)
    verbose = '--verbose' in sys.argv
    solver = SimplexSolver(c, A, b, senses, optimize, verbose)
    try:
        res = solver.solve()
        print(f"status: {res['status']}")
        print(f"objetivo: {res['objective']:.6f}")
        print(f"iteracoes: {res['iterations']}")
        print(f"tempo (s): {res['time']:.6f}")
    except ValueError as e:
        print(str(e))

if __name__ == '__main__':
    main()

