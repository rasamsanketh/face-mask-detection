let streaming = false;

function toggleStream() {
    fetch('/toggle_stream', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            streaming = data.streaming;
            if (streaming) {
                document.getElementById('video-feed').src = '/video_feed';
                startCounterUpdate();
            } else {
                document.getElementById('video-feed').src = '/static/video.png';
            }
        });
}

function toggleDarkMode(checkbox) {
    document.body.classList.toggle("dark-mode", checkbox.checked);
}

function startCounterUpdate() {
    const interval = setInterval(() => {
        if (!streaming) {
            clearInterval(interval);
            return;
        }
        fetch('/counts')
            .then(response => response.json())
            .then(data => {
                document.getElementById('mask-count').textContent = data.mask;
                document.getElementById('no-mask-count').textContent = data.no_mask;
            });
    }, 1000);
}
