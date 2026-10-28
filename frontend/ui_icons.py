"""Authored geometric interface icons; also exported into the offline atlas."""
ICONS = {'overview': '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>', 'questions': '<path d="M5 4h14v13H9l-4 4V4Z"/><path d="M10 8a2 2 0 1 1 3 1.7c-1 .6-1 1.3-1 2.3M12 14h.01"/>', 'graph': '<circle cx="5" cy="12" r="2.5"/><circle cx="18" cy="5" r="2.5"/><circle cx="18" cy="19" r="2.5"/><path d="m7.2 10.8 8.6-4.6M7.2 13.2l8.6 4.6"/>', 'library': '<path d="M4 4h5v16H4zM10 4h5v16h-5zM17 5l4 1-3 14-4-1"/>', 'archive': '<path d="M4 6h16v4H4zM6 10v10h12V10M10 14h4"/>', 'arrow': '<path d="M5 12h14m-5-5 5 5-5 5"/>', 'search': '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 5 5"/>', 'menu': '<path d="M5 6h14M5 12h14M5 18h14"/>', 'close': '<path d="m6 6 12 12M6 18 18 6"/>', 'book': '<path d="M3 5c4-1 6 0 9 2 3-2 5-3 9-2v15c-4-1-6 0-9 2-3-2-5-3-9-2V5Zm9 2v15"/>', 'code': '<path d="m8 6-5 6 5 6m8-12 5 6-5 6M14 4l-4 16"/>'}


def icon(name):
    return '<svg class="ui-icon" viewBox="0 0 24 24" aria-hidden="true">' + ICONS[name] + '</svg>'
