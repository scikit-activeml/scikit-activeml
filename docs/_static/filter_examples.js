$(document).ready(function () {
    // Only apply filtering to tables that are NOT marked "no-tag-filter"
    var $tables = $('.table').not('.no-tag-filter');

    var $filterableRows = $tables.find('tbody>tr'),
        $inputs = $('.input-tag');

    $tables.find('tr').each(function () {
        $(this).find('td').eq(2).hide();
        $(this).find('th').eq(2).hide();
    });
    $tables.find('colgroup').each(function () {
        $(this).find('col').eq(2).hide();
    });

    $inputs.on('input', function () {
        var selectedtags = [];
        $inputs.each(function () {
            if (this.checked) {
                selectedtags.push(this.value);
            }
        });

        var $matched = $filterableRows.hide().filter(function () {
            return $(this).find('td').eq(2).filter(function () {
                var tdText = $(this).text().toLowerCase();
                var matches = 0;
                selectedtags.forEach(function (item) {
                    if (tdText.indexOf(item) != -1) {
                        matches += 1;
                    }
                });
                return matches == selectedtags.length;
            }).length == 1;
        }).show();

        // A strategy has to carry every selected tag, so combining tags that
        // no strategy has in common empties the tables. Say so, because an
        // empty page otherwise looks like a broken filter.
        var count = $matched.length;
        var $status = $('#tag-filter-status');
        if (selectedtags.length === 0) {
            $status.text('');
            return;
        }
        if (count === 0) {
            $status.text(
                'No strategy has all of these properties: '
                + selectedtags.join(', ')
                + '. Try selecting fewer of them.'
            );
        } else {
            $status.text(
                count + (count === 1 ? ' strategy matches ' : ' strategies match ')
                + selectedtags.join(' + ') + '.'
            );
        }
    });
});
