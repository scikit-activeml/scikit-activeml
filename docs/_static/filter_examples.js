document.addEventListener('DOMContentLoaded', function () {
    const filters = document.getElementById('strategy-filters');
    if (!filters) {
        return;
    }

    const tables = Array.from(document.querySelectorAll('table.strategy-table'));
    const groups = Array.from(filters.querySelectorAll('[data-filter]'));
    const status = document.getElementById('tag-filter-status');
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

    function updateFilters() {
        const selected = {};
        groups.forEach(function (group) {
            selected[group.dataset.filter] = group.querySelector('input:checked').value;
        });

        const matched = entries.filter(function (entry) {
            // Match all target constraints within one supported setup.
            const supported = entry.metadata.capabilities.some(function (setup) {
                return (!selected.task || setup[0] === selected.task)
                    && (!selected.target || setup[1] === selected.target)
                    && (!selected.annotators || setup[2] === selected.annotators);
            });
            const batch = entry.metadata.batch;
            const batchMatches = !selected.batch || (selected.batch === 'batch'
                ? batch.some(value => value !== 'single-sample')
                : batch.includes(selected.batch));
            entry.row.hidden = !(supported && batchMatches);
            return !entry.row.hidden;
        });

        sections.forEach(function (item) {
            const empty = Array.from(item.table.tBodies[0].rows)
                .every(row => row.hidden);
            item.section.hidden = empty;
            item.tocItems.forEach(item => { item.hidden = empty; });
        });

        const count = matched.length;
        const classes = new Set(matched.map(entry => entry.metadata.api_class)).size;
        let summary = 'Showing ' + count + ' of ' + entries.length
            + ' entries across ' + classes + ' API '
            + (classes === 1 ? 'class.' : 'classes.');
        if (count === 0) {
            summary += ' No entries match these filters. '
                + 'Select Any in a filter group or choose Clear all.';
        }
        status.textContent = summary;
    }

    groups.forEach(group => group.addEventListener('change', updateFilters));
    document.getElementById('clear-strategy-filters').addEventListener('click', function () {
        groups.forEach(function (group) {
            group.querySelector('input[value=""]').checked = true;
        });
        updateFilters();
    });
    updateFilters();
    filters.hidden = false;
});
