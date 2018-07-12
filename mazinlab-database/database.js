


$(function() {
                $("[id^=20]").click(function() {
                    console.log(1);
   $.getJSON($(this).text()+"J.json", function(data) {
        var to_append = ""; // to store string of html to append to ul
        var div = $("#Runs"); // points to the 'div' (division) html element
        div.html(""); // clear all html contained by <div></div>

         $.each(data, function(key, val) {
           to_append += "<div class'container' id='"+key+"'><hr><p><h4>"+val["Target"]+
                             "</h2><b>Target: </b>"+val["Target"]+
                             "<br /><b>Type: </b>"+val["Type"]+
                             "<br /><b>J Mag: </b>"+val["J mag"]+
                             "<br /><b>Sunset Date(s): </b>"+val["Sunset Date(s)"]+
                             "<br /><b>Filters: </b>"+val["Filters"]+
                             "<br /><b>Time Windows: </b>"+val["Time Windows"]+
                              "<br /><b>Nearby Laser Cals: </b>"+val["Nearby Laser Cals"]+
                             "<br /><b>Important Notes: </b>"+val["Important Notes"]+
                             "<br /><b>Number of Dithers: </b>"+val["Number of Dithers"]+
                             "<br /><b>BIN File Range: </b>"+val["BIN File Range"]+"</p><hr></div>";
            });
         div.append(to_append);
       });
});
            });

  
    

    // Prevent 'enter' key from submitting forms (gives 404 error with full data set name form)
    $(window).keydown(function(event) {
        if (event.keyCode == 13) {
            event.preventDefault();
            return false;
        }
    });
