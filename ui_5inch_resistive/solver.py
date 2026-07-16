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
import math
import sympy
from sympy.parsing.latex import parse_latex

# Diagnostic output toggle. True while developing (prints the post-sanitize
# string, the branch taken, and full tracebacks); set False for a clean console
# on the Pi during a demo — the user-facing error strings are returned either way.
DEBUG = True


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


def _repair_grouping(s: str) -> str:
    """
    Best-effort repair of the structural/grouping mistakes the recognizer makes
    (dropped/added braces, doubled sub/superscript markers). Fixes SYNTAX only,
    so an otherwise-unparseable prediction can still be evaluated — it cannot
    recover intent when the model genuinely misread the structure.
    """
    # collapse doubled sub/superscript markers:  __ -> _ ,  ^^ -> ^
    s = re.sub(r'_{2,}', '_', s)
    s = re.sub(r'\^{2,}', '^', s)
    # drop a dangling trailing _ or ^ (no argument -> unparseable)
    s = re.sub(r'[_^]+\s*$', '', s)
    # balance braces: drop unmatched '}', append any still-open '}'.
    # Skip escaped braces (\{ \}, e.g. set notation) so they aren't miscounted.
    out, depth = [], 0
    for i, ch in enumerate(s):
        escaped = i > 0 and s[i - 1] == '\\'
        if ch == '{' and not escaped:
            depth += 1
            out.append(ch)
        elif ch == '}' and not escaped:
            if depth == 0:          # stray closing brace -> drop it
                continue
            depth -= 1
            out.append(ch)
        else:
            out.append(ch)
    return ''.join(out) + '}' * depth


def _safe_parse_latex(s: str):
    """
    parse_latex, applied to a structurally repaired version of the string.

    Repair runs proactively (not just on failure) because parse_latex often
    *silently* mis-parses broken grouping instead of raising — e.g. 'x^{2 + 1'
    parses to just 'x'. _repair_grouping is a no-op on well-formed LaTeX, so
    correct predictions are unaffected; only malformed ones change. Falls back
    to the raw string if the repair somehow parses worse.
    """
    repaired = _repair_grouping(s)
    if repaired == s:
        return parse_latex(s)
    try:
        return parse_latex(repaired)
    except Exception:
        return parse_latex(s)


def _clean(s: str) -> str:
    """Fix unicode chars that even DejaVu Sans Mono may lack."""
    return (s
        .replace('ⅈ', 'i')    # ⅈ imaginary unit → i
        .replace('ℯ', 'e')    # ℯ Euler's number → e
        .replace('⋅', '*')    # ⋅ dot product → *
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
# ── Matrix recovery helpers
# ---------------------------------------------------------------------------

def _sanitize_matrix_latex(s: str) -> str:
    """
    Strip predictable model output artifacts before parse_latex.
    """
    # Fix \bgin typo
    # -- use lookahead to avoid corrupting valid \begin
    s = s.replace(r'\bgin', r'\begin')
    s = re.sub(r'\\begi(?!n)', r'\\begin', s)

    # Normalize plain matrix → pmatrix for parse_latex compatibility
    s = re.sub(r'\\begin\{matrix\}', r'\\begin{pmatrix}', s)
    s = re.sub(r'\\end\{matrix\}', r'\\end{pmatrix}', s)

    # Strip leading [ (boundary token bleed from bmatrix training data)
    s = re.sub(r'^\[+', '', s.strip()).strip()

    # Strip trailing ]\ or ] artifacts (rstrip('\\') in run_solve already
    # handles bare trailing backslashes, so just strip the remaining ])
    s = re.sub(r'[\]\\]+$', '', s).strip()

    # Normalize 2+ backslashes before \end{ to single \end
    s = re.sub(r'\\{2,}end\{', r'\\end{', s)

    # Fix missing or corrupted \end{Xmatrix}
    begin_m = re.search(r'\\begin\{(\w*matrix)\}', s)
    if begin_m:
        env     = begin_m.group(1)
        end_tag = f'\\end{{{env}}}'
        if end_tag not in s:
            # Trim everything after the last clean matrix token
            # Valid interior chars: digits, letters, spaces, &, backslash
            s = re.sub(r'[^0-9a-zA-Z\s&\\]+$', '', s).strip()
            s = re.sub(r'[&\s]+$', '', s).strip()  # trim any dangling separator
            s = s + ' ' + end_tag

    # Collapse empty cells (&& → &0&) so parse_latex sees a valid entry
    s = re.sub(r'&&', '&0&', s)

    return s


def _recover_matrix(env_name: str, content: str) -> "sympy.Matrix | None":
    """
    Try to reconstruct a valid Matrix from garbled interior content
    where & and \\\\ separators are missing.
    Returns a sympy.Matrix on success, None on failure.
    """
    # If structural tokens are present, the problem is something else
    if '&' in content or '\\\\' in content:
        return None

    # Extract scalar tokens: integers, decimals, simple variable names
    tokens = re.findall(r'[+-]?\d+(?:\.\d+)?|[a-zA-Z]', content)
    if not tokens:
        return None

    n = len(tokens)
    sqrt_n = int(math.isqrt(n))

    if sqrt_n * sqrt_n == n and sqrt_n > 1:
        # Square matrix (2×2, 3×3, etc.)
        rows = [tokens[i * sqrt_n:(i + 1) * sqrt_n] for i in range(sqrt_n)]
    else:
        # Fall back to a column vector
        rows = [[t] for t in tokens]

    row_strs  = [' & '.join(row) for row in rows]
    recovered = (f'\\begin{{{env_name}}} '
                 + ' \\\\ '.join(row_strs)
                 + f' \\end{{{env_name}}}')

    try:
        return parse_latex(recovered)
    except Exception:
        return None


def _parse_matrix_direct(latex_str: str) -> sympy.Matrix:
    """
    Parse a matrix environment directly without parse_latex,
    which does not support \\begin{pmatrix} environments.
    Splits by \\\\ for rows and & for columns, then parses
    each scalar cell individually.
    """
    m = re.search(r'\\begin\{\w*matrix\}(.*?)\\end\{\w*matrix\}',
                  latex_str, re.DOTALL)
    if not m:
        raise ValueError('No matrix environment found')

    interior = m.group(1).strip()
    rows     = re.split(r'\\\\', interior)

    result = []
    for row in rows:
        row = row.strip()
        if not row:
            continue
        cells = row.split('&')
        result.append([
            _safe_parse_latex(re.sub(r'^0+(\d)', r'\1', c.strip()))
            if c.strip() else sympy.Integer(0)
            for c in cells
        ])

    if not result:
        raise ValueError('Matrix has no rows')

    # Pad ragged rows: a dropped & or \\ leaves rows uneven, which would make
    # sympy.Matrix() reject the whole thing. Best-effort — fill short rows with
    # zeros so a valid matrix can still be built from an imperfect prediction.
    max_cols = max(len(r) for r in result)
    for r in result:
        r.extend(sympy.Integer(0) for _ in range(max_cols - len(r)))

    return sympy.Matrix(result)


_MATRIX_BLOCK_RE = re.compile(r'\\begin\{\w*matrix\}.*?\\end\{\w*matrix\}', re.DOTALL)


def _parse_matrix_env(block: str) -> sympy.Matrix:
    """Parse one matrix environment, falling back to token recovery."""
    try:
        return _parse_matrix_direct(block)
    except Exception:
        m = re.search(r'\\begin\{(\w*matrix)\}(.*?)\\end\{\w*matrix\}', block, re.DOTALL)
        rec = _recover_matrix(m.group(1), m.group(2).strip()) if m else None
        if rec is None:
            raise
        return rec


def _matrices_and_rest(s: str):
    """Return (list of parsed Matrices, leftover text with the blocks removed)."""
    mats = [_parse_matrix_env(b) for b in _MATRIX_BLOCK_RE.findall(s)]
    rest = _MATRIX_BLOCK_RE.sub(' ', s).strip()
    return mats, rest


def _product(mats):
    result = mats[0]
    for m in mats[1:]:
        result = result * m
    return result


def _label_block(label: str, block: str) -> str:
    """Prefix 'label = ' onto a (possibly multi-line) pretty-printed block."""
    lines  = block.split('\n')
    prefix = f'{label} = '
    pad    = ' ' * len(prefix)
    return '\n'.join([prefix + lines[0]] + [pad + l for l in lines[1:]])


# ---------------------------------------------------------------------------
# ── Matrix branch
# ---------------------------------------------------------------------------

def _solve_matrix(latex_str: str) -> tuple[str | None, str | None]:
    # ── Equation form ─────────────────────────────────────────────────────
    if _has_equals(latex_str):
        parts   = re.split(r'(?<!\\)=', latex_str, maxsplit=1)
        lhs_str = parts[0].strip()
        rhs_str = parts[1].strip()

        try:
            lhs_mats, lhs_rest = _matrices_and_rest(lhs_str)
            rhs_mats, rhs_rest = _matrices_and_rest(rhs_str)
        except Exception as e:
            return None, f'Could not parse matrix equation: {e}'

        # Variable letters left over once the matrix blocks are removed — the
        # unknown in a system, e.g. the 'x' in 'A x = b'.
        lhs_vars = set(re.findall(r'[A-Za-z]', lhs_rest))
        rhs_vars = set(re.findall(r'[A-Za-z]', rhs_rest))

        # -- Linear system  A x = b  --------------------------------------
        # LHS = one or more matrices (their product is A) times an unknown;
        # RHS = a constant vector. Solve for the unknown vector.
        if lhs_mats and rhs_mats and lhs_vars and not rhs_vars:
            try:
                A = _product(lhs_mats)
                b = _product(rhs_mats)
            except Exception as e:
                return None, f'Incompatible matrices in the system: {e}'
            if b.rows != A.rows:
                return None, (f'Dimension mismatch: coefficient matrix has '
                              f'{A.rows} rows but the constant vector has {b.rows}.')
            var = sorted(lhs_vars)[0]
            try:
                x, params = A.gauss_jordan_solve(b)
            except ValueError:
                return 'No solution — the system is inconsistent.', None
            except Exception as e:
                return None, f'Could not solve the system: {e}'
            out = _label_block(var, _clean(sympy.pretty(x, use_unicode=True)))
            if params.rows:      # free parameters remain
                out += '\n\n(infinitely many solutions; free parameters shown)'
            return out, None

        # -- No unknown: evaluate both sides and compare ------------------
        if lhs_mats and rhs_mats and not lhs_vars and not rhs_vars:
            try:
                L, R = _product(lhs_mats), _product(rhs_mats)
            except Exception as e:
                return None, f'Could not evaluate the matrix expression: {e}'
            if L.shape != R.shape:
                return 'False — the two sides have different shapes.', None
            equal = sympy.simplify(L - R).is_zero_matrix
            return ('True' if equal else 'False'), None

        return None, 'Unsupported matrix equation form.'

    # ── Plain form: display one matrix, or multiply several ───────────────
    try:
        mats, _ = _matrices_and_rest(latex_str)
    except Exception as e:
        return None, f'Could not parse matrix: {e}'
    if not mats:
        return None, 'Could not find a matrix to parse.'
    try:
        return _clean(sympy.pretty(_product(mats), use_unicode=True)), None
    except Exception as e:
        return None, f'Matrix operation failed: {e}'


# ---------------------------------------------------------------------------
# ── Scalar branch
# ---------------------------------------------------------------------------

def _solve_scalar(latex_str: str) -> tuple[str | None, str | None]:
    if _has_equals(latex_str):
        parts   = re.split(r'(?<!\\)=', latex_str, maxsplit=1)
        lhs_str = parts[0].strip()
        rhs_str = parts[1].strip()

        try:
            lhs = _safe_parse_latex(lhs_str)
            rhs = _safe_parse_latex(rhs_str)
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
            expr = _safe_parse_latex(latex_str)
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

    # Catch \begin corruption variants before anything else touches the string.
    if r'\begin' in latex_str or r'\bgi' in latex_str:
        latex_str = _sanitize_matrix_latex(latex_str)

    latex_str = re.sub(r'(?<![\\a-zA-Z])([A-Z])(?![a-zA-Z])',
                       lambda m: m.group(1).lower(), latex_str)

    if DEBUG:
        print(f'[DEBUG] post-sanitize: {repr(latex_str)}')
        print(f'[DEBUG] is_matrix: {_is_matrix_expr(latex_str)}')

    try:
        if _is_matrix_expr(latex_str):
            return _solve_matrix(latex_str)
        else:
            return _solve_scalar(latex_str)
    except Exception as e:
        if DEBUG:
            import traceback; traceback.print_exc()
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
        # -- matrix cases (merged matrix handling) --
        (r'\begin{pmatrix}2 & 0\\0 & 2\end{pmatrix}', None),          # single matrix display
        (r'\begin{pmatrix}1 & 2\\3 & 4\end{pmatrix}'
         r'\begin{pmatrix}1 & 0\\0 & 1\end{pmatrix}', None),          # matrix multiplication
        # -- structural-repair cases (broken grouping the recognizer produces) --
        (r'x^{2 + 1',        None),   # dropped trailing '}'  -> x^{2+1}
        (r'x}^2 + 1',        None),   # stray '}'             -> x^2 + 1
        (r'x^^2 = 4',        None),   # doubled '^'           -> x^2 = 4
        # -- linear systems & matrix equations --
        (r'\begin{pmatrix}1&1\\0&1\end{pmatrix}x=\begin{pmatrix}3\\1\end{pmatrix}', None),  # A x = b
        (r'\begin{pmatrix}2&0\\0&2\end{pmatrix}x=\begin{pmatrix}4\\6\end{pmatrix}', None),  # diagonal system
        (r'\begin{pmatrix}1&0\\0&1\end{pmatrix}=\begin{pmatrix}1&0\\0&1\end{pmatrix}', None),  # equality (True)
        # -- ragged matrix (a dropped &) should still form a matrix via padding --
        (r'\begin{pmatrix}1&2&3\\4&5\end{pmatrix}', None),
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
