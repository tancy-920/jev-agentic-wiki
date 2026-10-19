"""Literal, bounded comparison navigation. No factual labels or answer oracle."""
import re


def comparison_grid(query):
    """Extract only an explicit Chinese object-list:comparison-axis-list form.

    Ambiguous prose, nested delimiters and oversized grids fall back to no grid.
    Every cell is a pair of exact query spans; cells are not new requirements or
    evidence. Model coverage observations remain fallible, never ground truth.
    """
    if not isinstance(query, str) or re.search(r'''["'‘’“”`()（）\[\]【】{}]''', query):
        return None
    match = re.fullmatch(r"\s*(?:请)?比较(?P<objects>[^：:？?。]+)[：:](?P<axes>[^：:？?。]+)[？?。]?\s*", query)
    if not match:
        return None
    objects = [v.strip() for v in re.split(r"、|和", match['objects'])]
    axes = [v.strip() for v in re.split(r"[，,、]|以及|及", match['axes'])]
    if not (2 <= len(objects) <= 6 and 2 <= len(axes) <= 5 and len(objects)*len(axes) <= 24):
        return None
    if any(not 1 <= len(v) <= 48 for v in objects+axes) or len(set(objects)) != len(objects) or len(set(axes)) != len(axes):
        return None
    # The scope marker explicitly distributes the requested axes over objects.
    if not any('各' in v or '分别' in v for v in axes):
        return None
    def spans(values, start):
        output = []
        for value in values:
            pos = query.find(value, start)
            output.append(dict(text=value, start=pos, end=pos+len(value)))
            start = pos+len(value)
        return output
    return dict(objects=spans(objects, match.start('objects')),
                axes=spans(axes, match.start('axes')),
                policy='Exact original-query navigation only. Check EACH object on EACH requested axis against the actual written answer and its original sources. No cell is satisfied merely by naming the object or citing a source. Do not invent an answer or an extra requirement.',
                semantic_support_certified=False, additional_provider_calls=0)


def comparison_check_schema(grid):
    keys = [f'o{i}' for i in range(len(grid['objects']))]
    return dict(type='object', additionalProperties=False,
                properties={k: dict(type='array', minItems=len(grid['axes']), maxItems=len(grid['axes']), items=dict(type='boolean')) for k in keys},
                required=keys)


def validate_comparison_checks(value, grid):
    keys = {f'o{i}' for i in range(len(grid['objects']))}
    return (isinstance(value, dict) and set(value) == keys
            and all(isinstance(row, list) and len(row) == len(grid['axes'])
                    and all(type(cell) is bool for cell in row) for row in value.values()))
