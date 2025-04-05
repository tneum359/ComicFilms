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

    // Comic Carousel Rotation Logic
    class ComicCarousel {
        constructor() {
            this.container = document.querySelector('.comic-container');
            this.comics = Array.from(document.querySelectorAll('.comic'));
            this.positions = ['position-1', 'position-2', 'position-3', 'position-4', 'position-5'];
            this.isAnimating = false;
            this.autoRotateInterval = 8000;
            this.autoRotateTimer = null;
            
            this.init();
        }
        
        init() {
            // Add arrow navigation
            const nav = document.createElement('div');
            nav.className = 'carousel-nav';
            
            const prevButton = document.createElement('button');
            prevButton.className = 'nav-button prev';
            prevButton.setAttribute('aria-label', 'Previous');
            
            const nextButton = document.createElement('button');
            nextButton.className = 'nav-button next';
            nextButton.setAttribute('aria-label', 'Next');
            
            nav.appendChild(prevButton);
            nav.appendChild(nextButton);
            this.container.parentElement.appendChild(nav);
            
            // Event listeners for arrows
            prevButton.onclick = () => {
                this.rotate('left');
                this.restartAutoRotate();
            };
            
            nextButton.onclick = () => {
                this.rotate('right');
                this.restartAutoRotate();
            };
            
            // Add click handlers to all comics
            this.comics.forEach((comic, index) => {
                comic.addEventListener('click', () => this.moveToCenter(index));
            });
            
            this.startAutoRotate();
        }
        
        moveToCenter(clickedIndex) {
            if (this.isAnimating) return;
            this.isAnimating = true;
            
            // Get current positions array
            const currentPositions = [...this.positions];
            
            // Find where the clicked comic currently is
            const clickedPosition = currentPositions[clickedIndex];
            const centerPosition = 'position-3';
            
            if (clickedPosition === centerPosition) {
                this.isAnimating = false;
                return; // Already centered
            }
            
            // Rearrange positions array to put clicked comic in center
            while (currentPositions[clickedIndex] !== centerPosition) {
                currentPositions.unshift(currentPositions.pop()); // Rotate right
            }
            
            // Apply new positions
            this.comics.forEach((comic, index) => {
                comic.style.transition = 'all 0.8s cubic-bezier(0.4, 0, 0.2, 1)';
                comic.className = `comic ${currentPositions[index]}`;
            });
            
            // Update positions array
            this.positions = currentPositions;
            
            // Reset animation flag and restart auto-rotate
            setTimeout(() => {
                this.isAnimating = false;
                this.restartAutoRotate();
            }, 800);
        }
        
        rotate(direction) {
            if (this.isAnimating) return;
            this.isAnimating = true;
            
            const newPositions = [...this.positions];
            if (direction === 'right') {
                newPositions.unshift(newPositions.pop());
            } else {
                newPositions.push(newPositions.shift());
            }
            
            this.comics.forEach((comic, index) => {
                comic.style.transition = 'all 0.8s cubic-bezier(0.4, 0, 0.2, 1)';
                comic.className = `comic ${newPositions[index]}`;
            });
            
            this.positions = newPositions;
            
            setTimeout(() => {
                this.isAnimating = false;
            }, 800);
        }
        
        startAutoRotate() {
            if (!this.autoRotateTimer) {
                this.autoRotateTimer = setInterval(() => {
                    this.rotate('right');
                }, this.autoRotateInterval);
            }
        }
        
        stopAutoRotate() {
            if (this.autoRotateTimer) {
                clearInterval(this.autoRotateTimer);
                this.autoRotateTimer = null;
            }
        }
        
        restartAutoRotate() {
            this.stopAutoRotate();
            this.startAutoRotate();
        }
    }

    // Initialize when DOM is loaded
    const carousel = new ComicCarousel();

    // Updated filtering functionality
    const tagCheckboxes = document.querySelectorAll('.tag-checkbox input');
    const comics = document.querySelectorAll('.grid-comic');
    
    function updateComicVisibility() {
        const selectedTags = Array.from(tagCheckboxes)
            .filter(checkbox => checkbox.checked)
            .map(checkbox => checkbox.value);
        
        comics.forEach(comic => {
            const comicTags = comic.dataset.tags.split(',');
            // Show comic if no tags are selected or if it matches any selected tag
            const shouldShow = selectedTags.length === 0 || 
                             selectedTags.some(tag => comicTags.includes(tag));
            comic.style.display = shouldShow ? 'block' : 'none';
        });
    }
    
    tagCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', updateComicVisibility);
    });
});
