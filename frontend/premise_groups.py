"""Exact source fragments in separate citation-set namespaces, no new evidence."""
from copy import deepcopy


def scoped_premise_groups(targets, original_view, original_inventory):
    result, inventory, views, groups = deepcopy(targets), {}, {}, {}
    covered = set()
    for target in result.values():
        sources = tuple(sorted(set(target['source_ids'])))
        if sources not in groups:
            if len(groups) >= 4:
                raise ValueError('too many distinct premise citation groups')
            group = f'G{len(groups)+1:02d}'
            groups[sources] = group
            fragments = {}
            count = 0
            for old_id, span in original_inventory.items():
                if span['chunk_id'] not in sources:
                    continue
                if len(inventory) >= 2048:
                    raise ValueError('scoped premise inventory exceeds bound')
                count += 1
                identifier = f'{group}-E{count:04d}'
                inventory[identifier] = dict(deepcopy(span), premise_group=group, original_excerpt_id=old_id)
                fragments.setdefault(span['chunk_id'], []).append(dict(excerpt_id=identifier, text=span['text']))
                covered.add(span['chunk_id'])
            views[group] = dict(source_ids=list(sources), complete_original_fragments=fragments)
        target['premise_group'] = groups[sources]
    # Keep original uncited contextual text, but do not offer it as a premise.
    views['uncited_context_only'] = {
        source: [dict(text=part['text']) for part in parts]
        for source, parts in original_view.items() if source not in covered}
    return result, views, inventory


def allowed_premise_ids(target, inventory):
    return [identifier for identifier, span in inventory.items()
            if span['chunk_id'] in target['source_ids']
            and span.get('premise_group') == target.get('premise_group')]
