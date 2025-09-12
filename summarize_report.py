import json
import argparse
import random
import sys
from typing import Dict, List, Tuple, Union


METRICS = ['correct', 'misplaced', 'missing', 'duplicates', 'extra']

metrics_dict = {
    'correct': 'Верные',
    'misplaced': 'Смещённые',
    'duplicates': 'Дубликаты',
    'extra': 'Лишние',
    'missing': 'Пропущенные'
}


def load_report(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_overall_cstats(report: Dict) -> Dict[str, int]:
    overall = report.get('overall', {})
    return overall.get('cstats', {})


def extract_per_situation_cstats(report: Dict) -> List[Dict[str, int]]:
    situations = report.get('situations', {})
    result = []
    for _, data in situations.items():
        cstats = data.get('cstats') if isinstance(data, dict) else None
        if isinstance(cstats, dict) and cstats.get('total_ref', 0) > 0:
            result.append(cstats)
    return result


def compute_point_estimates(cstats: Dict[str, int]) -> Dict[str, float]:
    total_ref = cstats.get('total_ref', 0)
    if total_ref <= 0:
        return {m: 0.0 for m in METRICS}
    return {m: cstats.get(m, 0) / total_ref for m in METRICS}


def aggregate_cstats(sample: List[Dict[str, int]]) -> Dict[str, int]:
    agg: Dict[str, int] = {}
    for cs in sample:
        for k, v in cs.items():
            agg[k] = agg.get(k, 0) + int(v)
    return agg


def bootstrap_cis(per_case_cstats: List[Dict[str, int]], iters: int, seed: Union[int, None], ci_pct: float) -> Dict[str, Tuple[float, float]]:
    if seed is not None:
        random.seed(seed)
    n = len(per_case_cstats)
    if n == 0:
        return {m: (None, None) for m in METRICS}

    alphas = (0.5 * (100 - ci_pct) / 100.0, 1 - 0.5 * (100 - ci_pct) / 100.0)
    samples: Dict[str, List[float]] = {m: [] for m in METRICS}

    for _ in range(iters):
        drawn = [per_case_cstats[random.randrange(n)] for _ in range(n)]
        agg = aggregate_cstats(drawn)
        total_ref = agg.get('total_ref', 0)
        if total_ref <= 0:
            # skip degenerate resample
            continue
        for m in METRICS:
            samples[m].append(agg.get(m, 0) / total_ref)

    def percentile(xs: List[float], q: float) -> float:
        if not xs:
            return None
        xs_sorted = sorted(xs)
        pos = (len(xs_sorted) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(xs_sorted) - 1)
        frac = pos - lo
        return xs_sorted[lo] * (1 - frac) + xs_sorted[hi] * frac

    ci: Dict[str, Tuple[float, float]] = {}
    for m in METRICS:
        lo = percentile(samples[m], alphas[0])
        hi = percentile(samples[m], alphas[1])
        ci[m] = (lo, hi)
    return ci


def to_json_output(point: Dict[str, float], cis: Dict[str, Tuple[float, float]], total_ref: int, iters: int, ci_pct: float) -> Dict:
    out = {
        'total_ref': total_ref,
        'ci': ci_pct,
        'bootstrap_iterations': iters,
        'metrics': {}
    }
    for m in METRICS:
        lo, hi = cis.get(m, (None, None))
        out['metrics'][m] = {
            'percent': point.get(m, 0.0),
            'ci_low': lo,
            'ci_high': hi
        }
    return out


def to_text_output(point: Dict[str, float], cis: Dict[str, Tuple[float, float]], total_ref: int, ci_pct: float) -> str:
    # Unicode table similar to the main script's style
    headers = ["Метрика", "Процент", f"ДИ {int(ci_pct)}%"]
    rows = []
    for m in METRICS:
        p = point.get(m, 0.0)
        lo, hi = cis.get(m, (None, None))
        ci_str = "—"
        if lo is not None and hi is not None:
            ci_str = f"{lo:.1%} – {hi:.1%}"
        rows.append([metrics_dict[m], f"{p:.1%}", ci_str])

    table = [headers] + rows
    col_widths = [max(len(str(cell)) for cell in col) for col in zip(*table)]

    h_line = "─"
    v_line = "│"
    tl_corner = "┌"
    tr_corner = "┐"
    bl_corner = "└"
    br_corner = "┘"
    t_down = "┬"
    t_up = "┴"
    t_right = "├"
    t_left = "┤"
    cross = "┼"

    out_lines = []
    # top border
    top = tl_corner
    for j, w in enumerate(col_widths):
        top += h_line * (w + 2)
        top += tr_corner if j == len(col_widths) - 1 else t_down
    out_lines.append(top)

    for i, row in enumerate(table):
        line = v_line
        for j, cell in enumerate(row):
            align = "<" if j == 0 else ">"
            line += f" {str(cell):{align}{col_widths[j]}} "
            line += v_line
        out_lines.append(line)
        # separator after header and at the end
        if i == 0:
            sep = t_right
            for j, w in enumerate(col_widths):
                sep += h_line * (w + 2)
                sep += t_left if j == len(col_widths) - 1 else cross
            out_lines.append(sep)

    bottom = bl_corner
    for j, w in enumerate(col_widths):
        bottom += h_line * (w + 2)
        bottom += br_corner if j == len(col_widths) - 1 else t_up
    out_lines.append(bottom)

    out_lines.append(f"Всего (total_ref): {total_ref}")
    return "\n".join(out_lines)


def to_html_output(point: Dict[str, float], cis: Dict[str, Tuple[float, float]], total_ref: int, ci_pct: float) -> str:
    html = [
        '<!DOCTYPE html>',
        '<html><head><meta charset="utf-8">',
        '<title>Сводная таблица метрик</title>',
        '<style>',
        'body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 900px; margin: 0 auto; padding: 20px; }',
        'table { border-collapse: collapse; width: 100%; }',
        'th, td { border: 1px solid #999; padding: .5em .6em; }',
        'th { background-color: #f2f2f2; }',
        '</style></head><body>',
        '<h1>Сводная таблица метрик</h1>',
        f'<p>Всего (total_ref): <strong>{total_ref}</strong>, доверительный интервал: {int(ci_pct)}%</p>',
        '<table>',
        '<tr><th>Метрика</th><th>Процент</th><th>ДИ</th></tr>'
    ]
    for m in METRICS:
        p = point.get(m, 0.0)
        lo, hi = cis.get(m, (None, None))
        if lo is not None and hi is not None:
            ci_str = f"{lo:.1%} – {hi:.1%}"
        else:
            ci_str = '—'
        html.append(f"<tr><td>{metrics_dict[m]}</td><td align='right'>{p:.1%}</td><td align='right'>{ci_str}</td></tr>")
    html.extend(['</table>', '</body></html>'])
    return "\n".join(html)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Сводка JSON-отчета: проценты по метрикам и bootstrap-ДИ',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('report', type=str, help='Путь к JSON-отчету (generate_multi_situation_report_json)')
    parser.add_argument('--format', '-f', choices=['json', 'text', 'html'], default='text', help='Формат вывода')
    parser.add_argument('--output', '-o', type=str, default='-', help='Файл вывода, либо - для stdout')
    parser.add_argument('--bootstrap-iters', type=int, default=2000, help='Число итераций бутстрэпа')
    parser.add_argument('--seed', type=int, default=None, help='Сид генератора случайных чисел')
    parser.add_argument('--ci', type=float, default=95.0, help='Уровень ДИ в процентах (например, 95)')

    args = parser.parse_args()

    report = load_report(args.report)
    overall_cstats = extract_overall_cstats(report)
    per_case = extract_per_situation_cstats(report)

    total_ref = int(overall_cstats.get('total_ref', 0))
    point = compute_point_estimates(overall_cstats)

    cis = bootstrap_cis(per_case, args.bootstrap_iters, args.seed, args.ci)

    if args.format == 'json':
        out_obj = to_json_output(point, cis, total_ref, args.bootstrap_iters, args.ci)
        out = json.dumps(out_obj, ensure_ascii=False, indent=2)
    elif args.format == 'html':
        out = to_html_output(point, cis, total_ref, args.ci)
    else:
        out = to_text_output(point, cis, total_ref, args.ci)

    if args.output == '-' or not args.output:
        print(out)
    else:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(out)
    return 0


if __name__ == '__main__':
    sys.exit(main()) 