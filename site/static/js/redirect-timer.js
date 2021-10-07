// timer and redirect
function startTimer(duration, display) {
    var timer = duration, seconds;
    var end =setInterval(function () {
        seconds = parseInt(timer % 60, 10);

        display.textContent = seconds;

        if (--timer < 0) {
            window.location = "/datumaro/docs";
            clearInterval(end);
        }
    }, 1000);
}

// timer display
window.onload = function () {
    var fiveSeconds = 5,
        display = document.querySelector('#time');
    startTimer(fiveSeconds, display);
};
