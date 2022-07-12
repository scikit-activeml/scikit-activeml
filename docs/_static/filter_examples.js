$(document).ready( function () {
    var $filterableRows = $('.table').find('tbody>tr'),
        $inputs = $('.input-tag');

    $('.table').find('tr').each(function () {
        $(this).find('td').eq(2).hide()
        $(this).find('th').eq(2).hide()
    });
    $('.table').find('colgroup').each(function () {
        $(this).find('col').eq(2).hide()
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