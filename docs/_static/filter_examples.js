document.addEventListener('DOMContentLoaded', function () {
    const filters = document.getElementById('strategy-filters');
    if (!filters) {
        return;
    }

    const tables = Array.from(document.querySelectorAll('table.strategy-table'));
    // Wrap API class names only where a new word starts.
    tables.forEach(function (table) {
        table.querySelectorAll('td:nth-child(2) a span').forEach(function (name) {
            const words = name.textContent
                .replace(/([a-z0-9])(?=[A-Z])|([A-Z])(?=[A-Z][a-z])/g, '$1$2\n')
                .split('\n');
            name.replaceChildren(...words.flatMap(function (word, index) {
                return index ? [document.createElement('wbr'), word] : [word];
            }));
        });
    });
    const groups = Array.from(filters.querySelectorAll('[data-filter]'));
    const status = document.getElementById('tag-filter-status');
    const summary = document.getElementById('strategy-filter-summary');
    const toggle = document.getElementById('toggle-strategy-filters');
    const entries = tables.flatMap(function (table) {
        return Array.from(table.tBodies[0].rows, function (row) {
            return {row: row, metadata: JSON.parse(row.cells[2].textContent)};
        });
    });
    const sections = tables.map(function (table) {
        const section = table.closest('section');
        const tocItems = Array.from(document.querySelectorAll('.bd-toc-nav a'))
            .filter(link => link.getAttribute('href') === '#' + section.id)
            .map(link => link.closest('li'));
        return {table: table, section: section, tocItems: tocItems};
    });

    function matches(metadata, selected) {
        // Match all target constraints within one supported setup.
        const supported = metadata.capabilities.some(function (setup) {
            return (!selected.task || setup[0] === selected.task)
                && (!selected.target || setup[1] === selected.target)
                && (!selected.annotators || setup[2] === selected.annotators);
        });
        return supported
            && (!selected.principle || metadata.principle === selected.principle)
            && (!selected.batch || metadata.batch.includes(selected.batch));
    }

    function optionLabel(input) {
        return input.closest('label').querySelector('span').textContent
            .replace(/\u00ad/g, '');
    }

    function updateFilters() {
        const selected = {};
        groups.forEach(function (group) {
            selected[group.dataset.filter] = group.querySelector('input:checked').value;
        });

        const matched = entries.filter(function (entry) {
            entry.row.hidden = !matches(entry.metadata, selected);
            return !entry.row.hidden;
        });

        // Dim each option that would leave no entries with the other choices,
        // and describe it as such to assistive technology.
        groups.forEach(function (group) {
            group.querySelectorAll('input:not([value=""])').forEach(function (input) {
                const option = Object.assign({}, selected, {[group.dataset.filter]: input.value});
                const empty = !entries.some(entry => matches(entry.metadata, option));
                input.closest('label').classList.toggle('strategy-filter-empty', empty);
                if (empty) {
                    input.setAttribute('aria-describedby', 'strategy-filter-empty-note');
                } else {
                    input.removeAttribute('aria-describedby');
                }
            });
        });

        sections.forEach(function (item) {
            const empty = Array.from(item.table.tBodies[0].rows)
                .every(row => row.hidden);
            item.section.hidden = empty;
            item.tocItems.forEach(item => { item.hidden = empty; });
        });

        const choices = Array.from(filters.querySelectorAll('input:checked'))
            .filter(input => input.value)
            .map(optionLabel);
        summary.textContent = choices.length ? choices.join(' · ') : 'Any setup';

        const count = matched.length;
        const classes = new Set(matched.map(entry => entry.metadata.api_class)).size;
        let text = 'Showing ' + count + ' of ' + entries.length
            + ' entries across ' + classes + ' API '
            + (classes === 1 ? 'class.' : 'classes.');
        if (count === 0) {
            text += ' No entries match these filters. '
                + 'Select Any in a filter group or choose Clear all.';
        }
        status.textContent = text;

        // Keep the selection in the URL, so that links can open this view.
        const params = new URLSearchParams(window.location.search);
        groups.forEach(function (group) {
            const value = selected[group.dataset.filter];
            if (value) {
                params.set(group.dataset.filter, value);
            } else {
                params.delete(group.dataset.filter);
            }
        });
        const query = params.toString();
        try {
            history.replaceState(history.state, '', window.location.pathname
                + (query ? '?' + query : '') + window.location.hash);
        } catch (error) {
            // Browsers may refuse frequent updates. The filter still works.
        }
    }

    // Links such as strategy_overview.html?task=regression preselect options.
    const params = new URLSearchParams(window.location.search);
    groups.forEach(function (group) {
        const value = params.get(group.dataset.filter);
        const input = Array.from(group.querySelectorAll('input'))
            .find(input => value && input.value === value);
        if (input) {
            input.checked = true;
        }
    });

    groups.forEach(group => group.addEventListener('change', updateFilters));
    document.getElementById('clear-strategy-filters').addEventListener('click', function () {
        groups.forEach(function (group) {
            group.querySelector('input[value=""]').checked = true;
        });
        updateFilters();
    });
    toggle.addEventListener('click', function () {
        const collapsed = filters.classList.toggle('strategy-filters-collapsed');
        toggle.setAttribute('aria-expanded', String(!collapsed));
        toggle.textContent = collapsed ? 'Show filters' : 'Hide filters';
    });
    updateFilters();
    filters.hidden = false;
});
