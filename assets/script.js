document.addEventListener('DOMContentLoaded', function() {
    console.log("JavaScript script loaded from external file.");
    setTimeout(function() {
        document.addEventListener('keydown', function(event) {
            console.log('Key pressed:', event.key);

            const keyMapping = {
                'Enter': 'accept-btn',
                'Backspace': 'reject-btn',
                'ArrowRight': 'next-btn',
                'ArrowLeft': 'prev-btn'
            };

            const buttonId = keyMapping[event.key];
            if (buttonId) {
                const button = document.getElementById(buttonId);
                if (button) {
                    button.click();
                    console.log('Simulated click on:', buttonId);
                } else {
                    console.warn('Button not found for ID:', buttonId);
                }
            }
        });
    }, 500);  // Attende 500 ms per il caricamento completo
});
