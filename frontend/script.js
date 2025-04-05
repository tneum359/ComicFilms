document.addEventListener('DOMContentLoaded', function() {
    // 1. Comic Card Hover Effects
    document.querySelectorAll('.comic-card').forEach(card => {
        card.addEventListener('mouseenter', () => {
            card.style.transform = 'translateY(-5px)';
            card.style.transition = 'transform 0.3s ease';
        });
        
        card.addEventListener('mouseleave', () => {
            card.style.transform = 'translateY(0)';
        });
    });

    // 2. Search Functionality
    const searchButton = document.querySelector('button[data-llama-id="32"]');
    let searchOpen = false;
    let searchBar = null;

    searchButton.addEventListener('click', () => {
        if (!searchOpen) {
            searchBar = document.createElement('input');
            searchBar.type = 'text';
            searchBar.placeholder = 'Search films...';
            searchBar.className = 'search-input p-2 rounded mr-2';
            searchButton.parentNode.insertBefore(searchBar, searchButton);
            searchOpen = true;
            
            // Focus the search input
            searchBar.focus();
        } else {
            searchBar.remove();
            searchOpen = false;
        }
    });

    // 3. Dark Mode Toggle
    const darkModeButton = document.querySelector('.dark-mode-toggle');
    
    let isDarkMode = false;
    darkModeButton.addEventListener('click', () => {
        isDarkMode = !isDarkMode;
        document.body.classList.toggle('dark-mode');
        const root = document.documentElement;
        
        if (isDarkMode) {
            document.body.style.backgroundColor = 'var(--framix-dark)';
            document.body.style.color = 'var(--framix-dark-text)';
            root.style.setProperty('--framix-primary', '#fce3d4');
            root.style.setProperty('--framix-secondary', '#f9a68b');
        } else {
            document.body.style.backgroundColor = 'var(--framix-light)';
            document.body.style.color = 'var(--text-primary)';
            root.style.setProperty('--framix-primary', '#691b1e');
            root.style.setProperty('--framix-secondary', '#b82d28');
        }
        darkModeButton.innerHTML = isDarkMode ? '☀️' : '🌙';
    });

    // 4. Reading Progress Tracker
    const progressKey = 'filmWatchProgress';
    let watchProgress = JSON.parse(localStorage.getItem(progressKey) || '{}');

    document.querySelectorAll('.comic-card').forEach(card => {
        card.addEventListener('click', (e) => {
            const filmTitle = card.querySelector('.font-semibold').textContent;
            if (!watchProgress[filmTitle]) {
                watchProgress[filmTitle] = {
                    lastWatched: new Date().toISOString(),
                    progress: 0
                };
                localStorage.setItem(progressKey, JSON.stringify(watchProgress));
            }
        });
    });

    // Image loading
    const images = document.querySelectorAll('img');
    images.forEach(img => {
        img.addEventListener('load', function() {
            this.classList.add('loaded');
        });
        if (img.complete) {
            img.classList.add('loaded');
        }
    });
});
