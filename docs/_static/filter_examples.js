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

        $filterableRows.hide().filter(function () {
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
    });
});
