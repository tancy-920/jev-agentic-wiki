"""Bounded complete-article graph candidate; not enabled in product routing."""

from collections import deque

from jev_rag.llm_wiki import alias_match, facet_tokens, tokens

ARMS = ("wiki_direct6", "wiki_one_hop", "wiki_three_hop")


def rank_direct(store, query):
    terms = tokens(query)
    ranked = []
    for identifier, row in store.pages.items():
        title = terms & store.title_tokens[identifier]
        body = terms & store.content_tokens[identifier]
        exact = any(alias_match(a, query) for a in [row.title, *row.aliases])
        if exact or title or len(body) >= 2:
            score = store.topical_score(identifier, terms) + 3 * exact
            ranked.append((score, identifier))
    return [identifier for _, identifier in sorted(ranked, key=lambda r: (-r[0], r[1]))]


def retrieve(store, queries, original, arm, *, removed_edges=()):
    if arm not in ARMS or not isinstance(queries, (list, tuple)) or not 1 <= len(queries) <= 2:
        raise ValueError("bounded experimental arm/queries required")
    if (not isinstance(original, str) or not original.strip() or len(original) > 6000
            or any(not isinstance(q, str) or not q.strip() or len(q) > 6000 for q in queries)):
        raise ValueError("bounded article query required")
    removed = set(removed_edges)
    if any(a not in store.pages or b not in store.pages[a].related for a, b in removed):
        raise ValueError("only existing edges may be removed")
    query = " ".join([original, *queries])
    direct = rank_direct(store, query)
    seeds = direct[:4]
    depth = {"wiki_direct6": 0, "wiki_one_hop": 1, "wiki_three_hop": 3}[arm]
    paths = {i: [i] for i in seeds}
    queue = deque(seeds)
    traversed = []
    while queue:
        node = queue.popleft()
        if len(paths[node]) - 1 >= depth:
            continue
        for neighbor in sorted(store.pages[node].related):
            if (node, neighbor) in removed:
                continue
            traversed.append([node, neighbor])
            if neighbor not in paths:
                paths[neighbor] = [*paths[node], neighbor]
                queue.append(neighbor)
    seed_facets = set().union(*(facet_tokens(" ".join([store.pages[i].title,
        *store.pages[i].aliases])) for i in seeds))
    requested = facet_tokens(query) - seed_facets
    candidates = []
    for identifier, path in paths.items():
        if identifier in seeds:
            continue
        matches = sorted(requested & store.content_facets[identifier])
        score = (2 * store.topical_score(identifier, tokens(original))
                 + store.topical_score(identifier, tokens(query))) / (1 + .15 * (len(path)-2))
        candidates.append(dict(article_id=identifier, path=path, distance=len(path)-1,
            score=score, matched_facets=matches, eligible=bool(matches) and score > 0))
    candidates.sort(key=lambda c: (-c["score"], c["article_id"]))
    neighbors = [c["article_id"] for c in candidates if c["eligible"]][:2]
    order = direct[:6] if depth == 0 else [*seeds, *neighbors]
    selected, omitted, size = [], [], 0
    for identifier in order:
        if size + len(store.texts[identifier]) > 24000:
            omitted.append(identifier)
        else:
            selected.append(identifier)
            size += len(store.texts[identifier])
    hits = [{"chunk": store.chunks["wiki-article:"+i].model_dump()} for i in selected]
    trace = dict(stage="experimental_bounded_article_paths", arm=arm, seeds=seeds,
        direct_ranking=direct, graph_neighbors=neighbors if depth else [], selected=selected,
        omitted_whole=omitted, graph_candidates=candidates, traversed_edges=traversed,
        removed_edges=sorted(removed), max_hops=depth, max_articles=6, article_characters=size,
        complete_articles=True, dense_search_calls=0, reranker_calls=0, new_provider_calls=0,
        paths_are_navigation_not_entailment=True)
    return hits, [trace]
