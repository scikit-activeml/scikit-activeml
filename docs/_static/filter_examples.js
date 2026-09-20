$(document).ready(function () {
    var $filters = $('#strategy-filters');
    if ($filters.length === 0) {
        return;
    }

    // Only apply filtering to tables that are NOT marked "no-tag-filter"
    var $tables = $('.table').not('.no-tag-filter');

    var $filterableRows = $tables.find('tbody>tr'),
        $inputs = $filters.find('.input-tag'),
        $status = $('#tag-filter-status');

    $tables.find('tr').each(function () {
        $(this).find('td').eq(2).hide();
        $(this).find('th').eq(2).hide();
    });
    $tables.find('colgroup').each(function () {
        $(this).find('col').eq(2).hide();
    });

    function updateFilters() {
        var selectedTags = [];
        $inputs.each(function () {
            if (this.checked) {
                selectedTags.push(this.value);
            }
        });

        var $matched = $filterableRows.hide().filter(function () {
            var tags = $(this).find('td').eq(2).text().toLowerCase().split(/\s+/);
            return selectedTags.every(function (tag) {
                return tags.indexOf(tag) !== -1;
            });
        }).show();

        // Explain empty results when no strategy matches all selected properties.
        var count = $matched.length;
        var summary = 'Showing ' + count + ' of ' + $filterableRows.length
            + ' strategies.';
        if (count === 0) {
            $status.text(
                summary + ' No strategy has all of these properties. '
                + 'Try selecting fewer properties or clear all filters.'
            );
        } else {
            $status.text(summary);
        }
    }

    $inputs.on('change', updateFilters);
    $('#clear-strategy-filters').on('click', function () {
        $inputs.prop('checked', false);
        updateFilters();
    });
    updateFilters();
});
