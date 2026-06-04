"""
solver.py  —  LaTeX expression solver
======================================
Handles:
  - Single equations with variables  (solve for unknowns)
  - Plain expressions with variables  (simplify)
  - Plain expressions, numbers only   (evaluate numerically)
  - Matrix multiplication             (A @ B)
  - Linear system solving             (A @ x = b  →  x)

Result strings use sympy.pretty() with unicode and preserve newlines.
The UI renders each line separately using a monospace font so that
fraction bars, roots, and matrix boxes align correctly.

Dependencies:
    pip install sympy antlr4-python3-runtime==4.11.1
"""

import re
import sympy
from sympy.parsing.latex import parse_latex


# ---------------------------------------------------------------------------
# ── Helpers
# ---------------------------------------------------------------------------

_MATRIX_RE = re.compile(
    r'\\begin\{(pmatrix|bmatrix|vmatrix|matrix|Bmatrix|Vmatrix)\}')


def _is_matrix_expr(latex_str: str) -> bool:
    return bool(_MATRIX_RE.search(latex_str))


def _has_equals(latex_str: str) -> bool:
    cleaned = re.sub(r'\\[a-zA-Z]+eq', '', latex_str)
    return '=' in cleaned


def _clean(s: str) -> str:
    """Fix unicode chars that even DejaVu Sans Mono may lack."""
    return (s
        .replace('\u2148', 'i')    # ⅈ imaginary unit → i
        .replace('\u212f', 'e')    # ℯ Euler's number → e
        .replace('\u22c5', '*')    # ⋅ dot product → *
    )


def _fmt(expr) -> str:
    """
    Pretty-print a single SymPy expression.
    Returns a (possibly multiline) string suitable for monospace rendering.
    """
    return _clean(sympy.pretty(expr, use_unicode=True))


def _format_solution(sol) -> str:
    """
    Format a SymPy solution into a multiline string.

    Each solution is on its own block separated by a blank line.
    For multiline values (fractions, roots), the variable label is
    aligned to the first line and subsequent lines are indented:

        x = -1

             √3·i
        x =  ────  +  1/2
              2
    """
    if isinstance(sol, dict):
        parts = []
        for k, v in sol.items():
            v_lines = _fmt(v).split('\n')
            k_str   = str(k)
            prefix  = f'{k_str} = '
            pad     = ' ' * len(prefix)
            if len(v_lines) == 1:
                parts.append(f'{prefix}{v_lines[0]}')
            else:
                lines = [prefix + v_lines[0]]
                lines += [pad + l for l in v_lines[1:]]
                parts.append('\n'.join(lines))
        return '\n\n'.join(parts)

    if isinstance(sol, (list, tuple)):
        return '\n\n'.join(_format_solution(s) for s in sol)

    # Bare value (e.g. from a one-symbol solve returning a list of values)
    return _fmt(sol)


# ---------------------------------------------------------------------------
# ── Matrix branch
# ---------------------------------------------------------------------------

def _solve_matrix(latex_str: str) -> tuple[str | None, str | None]:
    if _has_equals(latex_str):
        parts    = re.split(r'(?<!\\)=', latex_str, maxsplit=1)
        lhs_str  = parts[0].strip()
        rhs_str  = parts[1].strip()

        try:
            lhs = parse_latex(lhs_str)
            rhs = parse_latex(rhs_str)
        except Exception as e:
            return None, f'Could not parse matrix equation: {e}'

        if isinstance(lhs, sympy.MatMul):
            args     = lhs.args
            matrices = [a for a in args if isinstance(a, sympy.Matrix)]
            symbols  = [a for a in args if not isinstance(a, sympy.Matrix)]
            if len(matrices) == 1 and len(symbols) == 1:
                A = matrices[0]
                b = rhs if isinstance(rhs, sympy.Matrix) else sympy.Matrix([rhs])
                try:
                    x = A.solve(b)
                    return _clean(sympy.pretty(x, use_unicode=True)), None
                except Exception as e:
                    return None, f'System has no unique solution: {e}'

        try:
            eq   = sympy.Eq(lhs, rhs)
            free = eq.free_symbols
            if not free:
                diff = sympy.simplify(lhs - rhs)
                return ('True' if diff == 0 else 'False'), None
            sol = sympy.solve(eq, list(free), dict=True)
            if not sol:
                return 'No solution', None
            return _format_solution(sol), None
        except Exception as e:
            return None, f'Could not solve matrix equation: {e}'

    else:
        full_blocks = re.findall(
            r'\\begin\{(?:p|b|v|B|V)?matrix\}.*?\\end\{(?:p|b|v|B|V)?matrix\}',
            latex_str, re.DOTALL)

        if len(full_blocks) < 2:
            try:
                mat = parse_latex(latex_str)
                return _clean(sympy.pretty(mat, use_unicode=True)), None
            except Exception as e:
                return None, f'Could not parse matrix: {e}'

        try:
            mats = [parse_latex(b) for b in full_blocks]
        except Exception as e:
            return None, f'Could not parse matrices: {e}'

        try:
            result = mats[0]
            for m in mats[1:]:
                result = result * m
            return _clean(sympy.pretty(result, use_unicode=True)), None
        except Exception as e:
            return None, f'Matrix multiplication failed: {e}'


# ---------------------------------------------------------------------------
# ── Scalar branch
# ---------------------------------------------------------------------------

def _solve_scalar(latex_str: str) -> tuple[str | None, str | None]:
    if _has_equals(latex_str):
        parts   = re.split(r'(?<!\\)=', latex_str, maxsplit=1)
        lhs_str = parts[0].strip()
        rhs_str = parts[1].strip()

        try:
            lhs = parse_latex(lhs_str)
            rhs = parse_latex(rhs_str)
        except Exception as e:
            return None, f'Could not parse: {e}'

        free = (lhs - rhs).free_symbols
        if not free:
            diff = sympy.simplify(lhs - rhs)
            return ('True' if diff == 0 else 'False'), None

        eq  = sympy.Eq(lhs, rhs)
        sol = sympy.solve(eq, list(free), dict=True)
        if not sol:
            return 'No solution', None
        return _format_solution(sol), None

    else:
        try:
            expr = parse_latex(latex_str)
        except Exception as e:
            return None, f'Could not parse: {e}'

        free = expr.free_symbols
        if free:
            result = sympy.simplify(expr)
            return _clean(sympy.pretty(result, use_unicode=True)), None
        else:
            result = sympy.simplify(expr)
            if result.is_integer:
                return str(int(result)), None
            return _clean(sympy.pretty(sympy.nsimplify(result), use_unicode=True)), None


# ---------------------------------------------------------------------------
# ── Public API
# ---------------------------------------------------------------------------

def run_solve(inferred_latex: str) -> tuple[str | None, str | None]:
    """
    Compute a result from the inferred LaTeX string.
    Returns (result_str, error_str) — one is always None.
    result_str may contain newlines; render each line separately
    using a monospace font for correct alignment.
    """
    if not inferred_latex or not inferred_latex.strip():
        return None, 'No expression to solve.'

    latex_str = inferred_latex.strip()
    latex_str = latex_str.rstrip('\\').strip()
    latex_str = re.sub(r'(?<![\\a-zA-Z])([A-Z])(?![a-zA-Z])',
                       lambda m: m.group(1).lower(), latex_str)

    try:
        if _is_matrix_expr(latex_str):
            return _solve_matrix(latex_str)
        else:
            return _solve_scalar(latex_str)
    except Exception as e:
        return None, f'Solver error: {e}'


# ---------------------------------------------------------------------------
# ── Smoke tests
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    tests = [
        (r'2 + 4',           '6'),
        (r'\frac{10}{2}',    '5'),
        (r'x^2 + 2x + x^2',  None),
        (r'x^2 + 5x + 6 = 0', None),
        (r'2x + 4 = 0',       None),
        (r'x^3 = -1',         None),
    ]

    print('Running solver smoke tests…\n')
    all_ok = True
    for latex, expected in tests:
        result, err = run_solve(latex)
        status = 'ERR' if err else 'OK'
        if expected and result != expected:
            status = 'FAIL'
            all_ok = False
        print(f'  [{status}]  {latex}')
        if result:
            for line in result.split('\n'):
                print(f'         {line}')
        else:
            print(f'         ERROR: {err}')
        print()

    print('All tests passed.' if all_ok else 'Some tests failed.')