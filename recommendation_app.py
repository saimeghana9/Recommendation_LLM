import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import os
import re
from difflib import get_close_matches

# Set page config
st.set_page_config(
    page_title="Cross-Domain Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
    }
    .recommendation-card {
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
        background-color: #f9f9f9;
    }
    .stSpinner > div {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load data from GitHub or local directory
@st.cache_data
def load_data():
    # GitHub repository details
    github_user = "saimeghana9"
    github_repo = "Recommendation_LLM"
    github_branch = "main"
    
    # First try to load from GitHub
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Loading movies data from GitHub...")
        movies_url = f"https://raw.githubusercontent.com/{github_user}/{github_repo}/{github_branch}/movies.csv"
        movies_df = pd.read_csv(movies_url)
        progress_bar.progress(20)
        
        status_text.text("Loading books data from GitHub...")
        books_url = f"https://raw.githubusercontent.com/{github_user}/{github_repo}/{github_branch}/books.csv"
        books_df = pd.read_csv(books_url)
        progress_bar.progress(40)
        
        status_text.text("Loading food data from GitHub...")
        food_url = f"https://raw.githubusercontent.com/{github_user}/{github_repo}/{github_branch}/food.csv"
        food_df = pd.read_csv(food_url)
        progress_bar.progress(60)
        
        status_text.text("Loading music data from GitHub...")
        music_url = f"https://raw.githubusercontent.com/{github_user}/{github_repo}/{github_branch}/music.csv"
        music_df = pd.read_csv(music_url)
        progress_bar.progress(80)
        
        status_text.text("Loading TV shows data from GitHub...")
        tv_shows_url = f"https://raw.githubusercontent.com/{github_user}/{github_repo}/{github_branch}/tv_shows.csv"
        tv_shows_df = pd.read_csv(tv_shows_url)
        progress_bar.progress(100)
        
        status_text.text("Data loaded successfully from GitHub!")
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()
        
        return movies_df, books_df, food_df, music_df, tv_shows_df
        
    except Exception as e:
        st.warning(f"Could not load data from GitHub: {e}")
        st.info("Trying local directory...")
        
        # If GitHub fails, try local directory
        local_path = r"C:\Users\saime\Downloads\Rec"
        
        if not os.path.exists(local_path):
            st.error(f"Local directory not found: {local_path}")
            st.info("Falling back to sample data")
            return create_sample_data()
        
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Loading movies data from local...")
            movies_df = pd.read_csv(os.path.join(local_path, 'movies.csv'))
            progress_bar.progress(20)
            
            status_text.text("Loading books data from local...")
            books_df = pd.read_csv(os.path.join(local_path, 'books.csv'))
            progress_bar.progress(40)
            
            status_text.text("Loading food data from local...")
            food_df = pd.read_csv(os.path.join(local_path, 'food.csv'))
            progress_bar.progress(60)
            
            status_text.text("Loading music data from local...")
            music_df = pd.read_csv(os.path.join(local_path, 'music.csv'))
            progress_bar.progress(80)
            
            status_text.text("Loading TV shows data from local...")
            tv_shows_df = pd.read_csv(os.path.join(local_path, 'tv_shows.csv'))
            progress_bar.progress(100)
            
            status_text.text("Data loaded successfully from local directory!")
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()
            
            return movies_df, books_df, food_df, music_df, tv_shows_df
            
        except Exception as e:
            st.error(f"Error loading local data: {e}")
            st.info("Falling back to sample data")
            return create_sample_data()

def create_sample_data():
    """Create sample data for demonstration if CSV files are not available"""
    # Sample movies data
    movies_data = {
        'title': ['The Shawshank Redemption', 'The Godfather', 'The Dark Knight', 
                 'Pulp Fiction', 'Forrest Gump', 'Inception', 'The Matrix'],
        'genre': ['Drama', 'Crime', 'Action', 'Crime', 'Drama', 'Sci-Fi', 'Action'],
        'mood': ['Inspiring', 'Intense', 'Thrilling', 'Edgy', 'Heartwarming', 'Mind-bending', 'Exciting'],
        'keywords': ['prison hope redemption', 'mafia family power', 'superhero villain chaos',
                    'crime nonlinear storytelling', 'life journey love', 'dreams reality layers',
                    'simulation action philosophy'],
        'rating': [9.3, 9.2, 9.0, 8.9, 8.8, 8.8, 8.7],
        'description': [
            'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.',
            'The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.',
            'When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.',
            'The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.',
            'The presidencies of Kennedy and Johnson, the events of Vietnam, Watergate, and other historical events unfold through the perspective of an Alabama man with an IQ of 75.',
            'A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.',
            'A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.'
        ]
    }
    
    # Sample books data
    books_data = {
        'title': ['To Kill a Mockingbird', '1984', 'Pride and Prejudice', 
                 'The Great Gatsby', 'The Hobbit', 'The Catcher in the Rye'],
        'author': ['Harper Lee', 'George Orwell', 'Jane Austen', 
                  'F. Scott Fitzgerald', 'J.R.R. Tolkien', 'J.D. Salinger'],
        'genre': ['Fiction', 'Dystopian', 'Romance', 'Fiction', 'Fantasy', 'Fiction'],
        'mood': ['Thought-provoking', 'Dark', 'Romantic', 'Tragic', 'Adventurous', 'Coming-of-age'],
        'keywords': ['racism justice childhood', 'totalitarianism surveillance rebellion', 
                    'love class society', 'american dream jazz age', 'quest fantasy adventure',
                    'teenage angst identity'],
        'average_rating': [4.7, 4.6, 4.5, 4.3, 4.8, 4.2],
        'description': [
            'The story of young Scout Finch and her father, a lawyer who defends a black man accused of raping a white woman in the Depression-era South.',
            'A dystopian social science fiction novel that examines the consequences of totalitarianism, mass surveillance, and repressive regimentation.',
            'A romantic novel of manners that depicts the emotional development of protagonist Elizabeth Bennet.',
            'A story of Jay Gatsby, a self-made millionaire, and his pursuit of Daisy Buchanan, a wealthy young woman whom he loved in his youth.',
            'A fantasy novel about the adventures of hobbit Bilbo Baggins, who is hired as a burglar by a group of dwarves on a quest to reclaim their mountain home from a dragon.',
            'A story about Holden Caulfield and his experiences in New York City after being expelled from prep school.'
        ]
    }
    
    # Sample food data
    food_data = {
        'name': ['Spaghetti Carbonara', 'Chicken Tikka Masala', 'Vegetable Stir Fry', 
                'Chocolate Chip Cookies', 'Avocado Toast', 'Greek Salad'],
        'cuisine_type': ['Italian', 'Indian', 'Asian', 'American', 'International', 'Greek'],
        'mood': ['Comforting', 'Spicy', 'Healthy', 'Sweet', 'Fresh', 'Refreshing'],
        'keywords': ['pasta bacon egg cheese', 'chicken creamy tomato spicy', 'vegetables quick healthy',
                    'chocolate sweet baked', 'avocado bread simple', 'cucumber tomato feta'],
        'rating': [4.8, 4.5, 4.2, 4.7, 4.0, 4.3],
        'ingredients': ['Spaghetti, eggs, cheese, pancetta, black pepper', 
                       'Chicken, yogurt, spices, tomato sauce, cream',
                       'Mixed vegetables, soy sauce, garlic, ginger, oil',
                       'Flour, butter, sugar, chocolate chips, eggs',
                       'Bread, avocado, salt, pepper, olive oil',
                       'Cucumber, tomato, red onion, feta cheese, olives, olive oil'],
        'description': [
            'A classic Italian pasta dish with a creamy egg-based sauce, pancetta, and cheese.',
            'A popular Indian dish featuring grilled chicken in a spiced tomato and cream sauce.',
            'A quick and healthy dish with fresh vegetables stir-fried with Asian flavors.',
            'Classic homemade cookies with chunks of chocolate throughout.',
            'Simple yet delicious toast topped with mashed avocado and seasonings.',
            'A refreshing salad with Mediterranean ingredients and a tangy dressing.'
        ]
    }
    
    # Sample music data
    music_data = {
        'title': ['Bohemian Rhapsody', 'Hotel California', 'Blinding Lights', 
                 'Shape of You', 'Sweet Child O\' Mine', 'Billie Jean'],
        'artist': ['Queen', 'Eagles', 'The Weeknd', 
                  'Ed Sheeran', 'Guns N\' Roses', 'Michael Jackson'],
        'genre': ['Rock', 'Rock', 'Pop', 'Pop', 'Rock', 'Pop'],
        'mood': ['Epic', 'Mysterious', 'Energetic', 'Catchy', 'Nostalgic', 'Iconic'],
        'keywords': ['opera rock epic', 'california hotel mystery', 'synthwave retro upbeat',
                    'pop catchy dance', 'rock guitar riff nostalgic', 'pop iconic dance'],
        'lyrics': [
            'Is this the real life? Is this just fantasy? Caught in a landslide...',
            'On a dark desert highway, cool wind in my hair...',
            'I been tryna call, I been on my own for long enough...',
            'The club isn\'t the best place to find a lover...',
            'She\'s got a smile that it seems to me, reminds me of childhood memories...',
            'She was more like a beauty queen from a movie scene...'
        ]
    }
    
    # Sample TV shows data
    tv_shows_data = {
        'title': ['Breaking Bad', 'Game of Thrones', 'Friends', 
                 'Stranger Things', 'The Office', 'The Crown'],
        'genre': ['Drama', 'Fantasy', 'Comedy', 'Sci-Fi', 'Comedy', 'Drama'],
        'mood': ['Intense', 'Epic', 'Funny', 'Nostalgic', 'Quirky', 'Regal'],
        'keywords': ['chemistry crime transformation', 'fantasy politics dragons', 'friendship comedy relationships',
                    '80s supernatural mystery', 'workplace mockumentary comedy', 'royalty history drama'],
        'rating': [9.5, 9.2, 8.9, 8.7, 8.9, 8.6],
        'description': [
            'A high school chemistry teacher diagnosed with cancer turns to manufacturing and selling methamphetamine to secure his family\'s future.',
            'Nine noble families fight for control over the lands of Westeros, while an ancient enemy returns after being dormant for millennia.',
            'Follows the personal and professional lives of six twenty to thirty-something-year-old friends living in Manhattan.',
            'When a young boy vanishes, a small town uncovers a mystery involving secret experiments, terrifying supernatural forces and one strange little girl.',
            'A mockumentary on a group of typical office workers, where the workday consists of ego clashes, inappropriate behavior, and tedium.',
            'Follows the political rivalries and romance of Queen Elizabeth II\'s reign and the events that shaped the second half of the 20th century.'
        ]
    }
    
    return (
        pd.DataFrame(movies_data),
        pd.DataFrame(books_data),
        pd.DataFrame(food_data),
        pd.DataFrame(music_data),
        pd.DataFrame(tv_shows_data)
    )

class AdvancedRecommender:
    def __init__(self, movies_df, books_df, food_df, music_df, tv_shows_df):
        self.movies_df = movies_df
        self.books_df = books_df
        self.food_df = food_df
        self.music_df = music_df
        self.tv_shows_df = tv_shows_df
        
        # Create artist set for music filtering
        self.music_artists = set(self.music_df['artist'].str.lower().tolist())
        
        # For tracking recommendations to avoid duplicates
        self.recommended_items = {
            'movies': set(),
            'tv_shows': set(),
            'music': set(),
            'books': set(),
            'food': set()
        }
        
        # Prepare data
        self.prepare_domain_data()
        # Precompute TF-IDF models
        self.train_tfidf_models()
        
        # Common misspellings mapping
        self.common_misspellings = {
            'romcom': 'romcom',
            'romcoms': 'romcom',
            'romcom mobies': 'romcom movies',
            'romcom moveis': 'romcom movies',
            'romcom moives': 'romcom movies',
            'mobies': 'movies',
            'moveis': 'movies',
            'moives': 'movies',
            'muvi': 'movie',
            'muvies': 'movies',
            'bok': 'book',
            'boks': 'books',
            'recepie': 'recipe',
            'recipie': 'recipe',
            'reciepe': 'recipe',
            'musik': 'music',
            'muzik': 'music',
            'musick': 'music',
            'tvshow': 'tv show',
            'tvshows': 'tv shows',
            'television': 'tv'
        }
    
    def prepare_domain_data(self):
        """Prepare data for each domain with combined text features"""
        # Movies
        self.movies_df['combined_text'] = (
            self.movies_df['title'] + ' ' + 
            self.movies_df['genre'] + ' ' + 
            self.movies_df['mood'] + ' ' + 
            self.movies_df['keywords'] + ' ' +
            self.movies_df.get('director', '') + ' ' +
            self.movies_df.get('cast', '') + ' ' +
            self.movies_df.get('setting', '') + ' ' +
            self.movies_df.get('time_period', '')
        ).fillna('')
        
        # Books
        self.books_df['combined_text'] = (
            self.books_df['title'] + ' ' + 
            self.books_df['genre'] + ' ' + 
            self.books_df['mood'] + ' ' + 
            self.books_df['keywords'] + ' ' +
            self.books_df.get('author', '') + ' ' +
            self.books_df.get('setting', '') + ' ' +
            self.books_df.get('time_period', '')
        ).fillna('')
        
        # Food - Enhanced with more features
        self.food_df['combined_text'] = (
            self.food_df['name'] + ' ' + 
            self.food_df['cuisine_type'] + ' ' + 
            self.food_df['mood'] + ' ' + 
            self.food_df['keywords'] + ' ' +
            self.food_df['ingredients'] + ' ' +
            self.food_df.get('description', '') + ' ' +
            self.food_df.get('meal_type', '') + ' ' +
            self.food_df.get('dish_type', '') + ' ' +
            self.food_df.get('tags', '') + ' ' +
            self.food_df.get('category', '')
        ).fillna('')
        
        # Music
        self.music_df['combined_text'] = (
            self.music_df['title'] + ' ' + 
            self.music_df['artist'] + ' ' + 
            self.music_df['genre'] + ' ' + 
            self.music_df['mood'] + ' ' + 
            self.music_df['keywords'] + ' ' +
            self.music_df.get('album', '') + ' ' +
            self.music_df.get('year', '') + ' ' +
            self.music_df.get('instrumentation', '')
        ).fillna('')
        
        # TV Shows
        self.tv_shows_df['combined_text'] = (
            self.tv_shows_df['title'] + ' ' + 
            self.tv_shows_df['genre'] + ' ' + 
            self.tv_shows_df['mood'] + ' ' + 
            self.tv_shows_df['keywords'] + ' ' +
            self.tv_shows_df.get('creator', '') + ' ' +
            self.tv_shows_df.get('setting', '') + ' ' +
            self.tv_shows_df.get('time_period', '')
        ).fillna('')
    
    def train_tfidf_models(self):
        """Train TF-IDF models for each domain"""
        self.tfidf_vectorizers = {}
        self.tfidf_matrices = {}
        
        domains = {
            'movies': self.movies_df,
            'books': self.books_df,
            'food': self.food_df,
            'music': self.music_df,
            'tv_shows': self.tv_shows_df
        }
        
        for domain, df in domains.items():
            vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 3))
            tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
            self.tfidf_vectorizers[domain] = vectorizer
            self.tfidf_matrices[domain] = tfidf_matrix
    
    def correct_spelling(self, query):
        """Correct common spelling mistakes in the query"""
        query_lower = query.lower()
        
        # First, check for exact misspellings
        for misspelling, correction in self.common_misspellings.items():
            if misspelling in query_lower:
                query_lower = query_lower.replace(misspelling, correction)
        
        # Then use fuzzy matching for individual words
        words = query_lower.split()
        corrected_words = []
        
        for word in words:
            if len(word) <= 2:  # Skip very short words
                corrected_words.append(word)
                continue
                
            # Check if this word might be a misspelling of common domain terms
            domain_terms = ['movie', 'movies', 'film', 'book', 'books', 'music', 'song', 
                          'food', 'recipe', 'tv', 'show', 'shows', 'romcom', 'romantic', 'comedy']
            
            close_matches = get_close_matches(word, domain_terms, n=1, cutoff=0.7)
            if close_matches:
                corrected_words.append(close_matches[0])
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    def detect_domain(self, query: str):
        """Enhanced domain detection with spelling correction and single-word support"""
        # Correct spelling first
        corrected_query = self.correct_spelling(query)
        query_lower = corrected_query.lower()
        
        # Single word domain mapping
        single_word_domains = {
            'movies': ['movie', 'film', 'cinema', 'romcom', 'thriller', 'comedy', 'drama', 'action'],
            'tv_shows': ['tv', 'show', 'series', 'sitcom', 'kdrama'],
            'music': ['music', 'song', 'track', 'album', 'jazz', 'rock', 'pop'],
            'books': ['book', 'novel', 'read', 'fiction', 'fantasy', 'romance'],
            'food': ['food', 'recipe', 'dish', 'cooking', 'meal', 'pasta', 'pizza']
        }
        
        # Check for single word queries
        if len(query_lower.split()) == 1:
            for domain, words in single_word_domains.items():
                if query_lower in words:
                    return domain
            # If single word not found in mapping, default to movies for common entertainment terms
            if any(term in query_lower for term in ['movie', 'film', 'romcom']):
                return 'movies'
            elif any(term in query_lower for term in ['tv', 'show', 'series']):
                return 'tv_shows'
            elif any(term in query_lower for term in ['music', 'song']):
                return 'music'
            elif any(term in query_lower for term in ['book', 'read']):
                return 'books'
            elif any(term in query_lower for term in ['food', 'recipe']):
                return 'food'
        
        # Comprehensive domain mapping with extensive keyword matching
        domain_keywords = {
            'movies': [
                # General movie terms
                'movie', 'film', 'cinema', 'watch', 'thriller', 'funny', 'mysterious', 'romance', 'comedy', 'drama',
                'animated', 'holiday', 'courtroom', 'family', 'sports', 'sci-fi', 'tearjerker', 'classic',
                'time-travel', 'bollywood', 'realistic', 'iconic', 'cinephile', 'romcom','rom-com', 'notting hill',
                'inception', 'dark knight', 'black-and-white', 'slow-burn', 'feel-good', 'underrated',
                'powerful', 'inspirational', 'character depth', 'rewatch', 'award-winning', 'epic',
                'oscar', 'director', 'actor', 'actress', 'screenplay', 'plot', 'scene', 'sequel', 'prequel',
                # Specific genres and themes
                'action', 'adventure', 'fantasy', 'horror', 'mystery', 'suspense', 'crime', 'documentary',
                'biography', 'historical', 'war', 'western', 'musical', 'superhero', 'independent', 'foreign',
                'art house', 'blockbuster', 'cult classic', 'love story', 'romantic', 'plot', 'storyline',
                'great plots', 'love movies', 'funny', 'mysterious','action movies'
            ],
            'tv_shows': [
                'tv show', 'series', 'sitcom', 'k-drama', 'episode', 'season', 'binge', 'netflix', 'hulu',
                'hbo', 'streaming', 'mini-series', 'reality show', 'detective', 'medical drama', 'gilmore girls',
                'friends', 'game of thrones', 'breaking bad', 'sherlock', 'binge-worthy', 'twists',
                'character development', 'family-friendly', 'heartbreak', 'fantasy', 'limited series',
                'female leads', 'crime drama', 'animated series', 'wholesome', 'detective', 'medical',
                'high school', 'hidden gems', 'reality', 'tv', 'television', 'stream', 'watch'
            ],
            'music': [
                'music', 'song', 'track', 'album', 'jazz', 'rock', 'pop', 'lo-fi', 'lyrics', 'acoustic',
                'indie', 'classical', 'electronic', 'soundtrack', 'k-pop', 'meditation', 'piano', 'duet',
                'taylor swift', 'bts', 'cozy', 'iconic', 'upbeat', 'calm', 'powerful', 'studying', 'working out',
                'rainy days', 'underrated', 'golden classics', 'modern', 'electronic', 'classical', 'bollywood',
                'meditation', 'dance', 'live performances', 'soothing', 'nostalgic', 'road trip', 'mood lift',
                'artist', 'band', 'singer', 'composer', 'concert', 'playlist', 'genre', 'beat', 'rhythm', 'melody'
            ],
            'books': [
                'book', 'novel', 'read', 'fantasy', 'romance', 'historical', 'self-improvement', 'thriller',
                'biography', 'dystopian', 'short story', 'ya novel', 'classic', 'horror', 'philosophical',
                'gone girl', 'harry potter', 'hunger games', 'plot twist', 'character arcs', 'rich detail',
                'must-read', 'poetic', 'non-fiction', 'motivational', 'emotional depth', 'literary classics',
                'scary', 'female protagonists', 'light-hearted', 'philosophical', 'epic', 'trilogy', 'saga',
                'cozy', 'winter read', 'author', 'funny', 'mysterious','chapter', 'page', 'story', 'narrative', 'fiction', 'nonfiction'
            ],
            'food': [
                # General food terms
                'food', 'recipe', 'dish', 'cuisine', 'cooking', 'cook', 'meal', 'eat', 'dining', 'dinner',
                'lunch', 'breakfast', 'supper', 'snack', 'appetizer', 'main course', 'side dish', 'course',
                
                # Food types and categories
                'vegetarian', 'vegan', 'gluten-free', 'low-carb', 'keto', 'paleo', 'healthy', 'comfort food',
                'indulgent', 'gourmet', 'homemade', 'world cuisine', 'street food', 'iconic food', 'global cuisine',
                
                # Specific foods
                'taco', 'burger', 'pizza', 'noodles', 'sushi', 'pasta', 'rice', 'chicken', 'beef', 'pork',
                'seafood', 'fish', 'vegetable', 'fruit', 'salad', 'soup', 'stew', 'curry', 'sauce', 'dressing',
                'marinade', 'spread', 'dip', 
                
                # Cooking methods
                'bake', 'grill', 'fry', 'steam', 'roast', 'boil', 'simmer', 'saute', 'broil', 'barbecue', 'bbq',
                
                # Desserts and sweets
                'dessert', 'sweet', 'cake', 'pie', 'pastry', 'cookie', 'biscuit', 'brownie', 'pudding', 'custard',
                'ice cream', 'gelato', 'sorbet', 'chocolate', 'candy', 'confection', 'treat', 'bakery', 'baking',
                'muffin', 'cupcake', 'cheesecake', 'tiramisu', 'creme brulee', 'souffle', 'tart', 'donut', 'doughnut',
                
                # Drinks and beverages
                'drink', 'beverage', 'cocktail', 'smoothie', 'juice', 'coffee', 'tea', 'milkshake', 'soda', 'lemonade',
                'mocktail', 'shake', 'frappe', 'latte', 'cappuccino', 'espresso', 'brew', 'infusion', 'refresher',
                
                # Ingredients
                'egg', 'eggs', 'flour', 'sugar', 'butter', 'oil', 'spice', 'herb', 'garlic', 'onion', 'tomato',
                'cheese', 'milk', 'cream', 'yogurt', 'bread', 'grain', 'nut', 'seed', 'bean', 'lentil',
                
                # Cuisine types
                'italian', 'mexican', 'chinese', 'indian', 'japanese', 'french', 'thai', 'mediterranean', 'american',
                'fusion', 'spanish', 'greek', 'lebanese', 'vietnamese', 'korean', 'caribbean', 'brazilian',
                
                # Meal contexts
                'quick', 'easy', 'simple', 'fast', '30-minute', 'quick and easy', 'one-pot', 'one pan', 'sheet pan',
                'meal prep', 'make ahead', 'freezer friendly', 'batch cooking', 'party', 'gathering', 'celebration',
                'holiday', 'festive', 'special occasion', 'weeknight', 'weekend', 'brunch', 'picnic', 'potluck',
                
                # Descriptive terms
                'spicy', 'mild', 'savory', 'sweet', 'tangy', 'sour', 'bitter', 'umami', 'rich', 'light', 'fresh',
                'crispy', 'crunchy', 'creamy', 'chewy', 'tender', 'juicy', 'flavorful', 'aromatic', 'hearty',
                'refreshing', 'satisfying', 'comforting', 'wholesome', 'nutritious', 'decadent', 'elegant', 'rustic',
                
                # Specific queries
                'pasta recipes', 'pasta dish', 'pasta meal'
            ]
        }
        
        # Score each domain based on keyword matches with fuzzy matching
        domain_scores = {domain: 0 for domain in domain_keywords}
        
        for domain, keywords in domain_keywords.items():
            for keyword in keywords:
                # Use word boundaries to avoid partial matches
                if re.search(r'\b' + re.escape(keyword) + r'\b', query_lower):
                    domain_scores[domain] += 2  # Exact match gets higher score
                # Also check for partial matches with fuzzy matching
                elif len(keyword) > 3 and keyword in query_lower:
                    domain_scores[domain] += 1  # Partial match gets lower score
        
        # Find the domain with the highest score
        best_domain = max(domain_scores, key=domain_scores.get)
        
        # Only return a domain if it has at least one match, otherwise default to movies
        if domain_scores[best_domain] > 0:
            return best_domain
        else:
            # Default to movies for entertainment-related single words
            if len(query_lower.split()) == 1:
                return 'movies'
            return None
    
    def enhance_query(self, query, domain):
        """Enhance queries with related terms for better matching"""
        query_lower = query.lower()
        enhanced_query = query
        
        # Domain-specific query enhancers
        enhancers = {
            'movies': {
                'love': ['romance', 'romantic', 'relationship', 'heartfelt', 'emotional'],
                'action': ['adventure', 'thrilling', 'exciting', 'suspenseful', 'intense'],
                'great plots': ['story', 'narrative', 'plot twists', 'engaging', 'compelling'],
                'movies': ['film', 'cinema', 'motion picture', 'feature'],
                'romcom': ['romantic comedy', 'romance', 'comedy', 'love story']
            },
            'food': {
                'pasta': ['noodles', 'spaghetti', 'macaroni', 'penne', 'fettuccine', 'linguine'],
                'recipes': ['dish', 'meal', 'cooking', 'preparation'],
                'simple': ['easy', 'quick', 'basic', 'minimal', 'straightforward'],
                'impressive': ['elegant', 'fancy', 'gourmet', 'sophisticated', 'restaurant-quality']
            },
            'music': {
                'love': ['romantic', 'heartfelt', 'emotional', 'passionate'],
                'relaxing': ['calming', 'soothing', 'peaceful', 'tranquil'],
                'energetic': ['upbeat', 'lively', 'dynamic', 'vibrant']
            },
            'books': {
                'love': ['romance', 'relationship', 'heartfelt', 'emotional'],
                'thriller': ['suspense', 'mystery', 'crime', 'intrigue']
            },
            'tv_shows': {
                'drama': ['emotional', 'serious', 'intense', 'compelling'],
                'comedy': ['funny', 'humorous', 'lighthearted', 'entertaining']
            }
        }
        
        if domain in enhancers:
            for term, related_terms in enhancers[domain].items():
                if term in query_lower:
                    enhanced_query += ' ' + ' '.join(related_terms)
        
        return enhanced_query
    
    def process_query(self, query: str):
        """Process a user query and return recommendations"""
        # Detect the domain
        domain = self.detect_domain(query)
        
        if not domain:
            return "I can help with recommendations for movies, TV shows, music, books, and food. Please specify what you're looking for!"
        
        # Enhance the query with related terms
        enhanced_query = self.enhance_query(query, domain)
        
        # Special handling for music domain with artist filtering
        if domain == 'music':
            # Check if query contains any artist from our list
            found_artists = []
            for artist in self.music_artists:
                if re.search(r'\b' + re.escape(artist) + r'\b', query.lower()):
                    found_artists.append(artist)
            
            if found_artists:
                # Try to get recommendations from the specified artist first
                recs = self.get_recommendations(domain, enhanced_query, n_recommendations=10)
                # Filter to only include the requested artist
                artist_recs = recs[recs['artist'].str.lower().isin(found_artists)]
                if len(artist_recs) > 0:
                    return self._format_recommendations(artist_recs.head(3), domain, False)
                else:
                    # If no songs from the artist, fall back to general recommendations
                    recs = self.get_recommendations(domain, enhanced_query, n_recommendations=3)
                    return f"I couldn't find songs by that artist, but you might like these:\n\n" + \
                           self._format_recommendations(recs, domain, False)
        
        # Get recommendations for the detected domain
        recs = self.get_recommendations(domain, enhanced_query, 3)
        
        # Check if we found good matches
        if len(recs) == 0 or recs.iloc[0].get('similarity_score', 1) < 0.1:
            # Try a broader search if no good results found
            recs = self.get_recommendations(domain, query, 5)
            if len(recs) == 0:
                return f"Sorry, I couldn't find any {domain} recommendations for '{query}'. Try a different query!"
            else:
                # Found some similar items but not exact matches
                return f"I didn't find exact matches for '{query}', but here are some similar {domain} you might enjoy:\n\n" + \
                       self._format_recommendations(recs.head(3), domain, True)
        
        return self._format_recommendations(recs, domain, False)
    
    def get_recommendations(self, domain: str, query: str, n_recommendations: int = 3):
        """Get recommendations using TF-IDF and cosine similarity"""
        if domain not in self.tfidf_vectorizers:
            return pd.DataFrame()
        
        vectorizer = self.tfidf_vectorizers[domain]
        tfidf_matrix = self.tfidf_matrices[domain]
        
        query_vec = vectorizer.transform([query])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        # Get top N recommendations
        df = getattr(self, f"{domain}_df")
        top_indices = similarities.argsort()[::-1]  # Sort all indices by similarity
        
        # Filter out already recommended items
        unique_recs = []
        title_col = 'title' if domain != 'food' else 'name'
        
        for idx in top_indices:
            # Only consider items with reasonable similarity
            if similarities[idx] < 0.05:  # Lower threshold for broader matching
                continue
                
            item_id = df.iloc[idx][title_col]
            if item_id not in self.recommended_items[domain]:
                item = df.iloc[idx].copy()
                item['similarity_score'] = similarities[idx]  # Add similarity score
                unique_recs.append(item)
                self.recommended_items[domain].add(item_id)
            if len(unique_recs) >= n_recommendations:
                break
        
        # If we didn't find enough matches, try a broader search
        if len(unique_recs) < n_recommendations:
            for idx in top_indices:
                item_id = df.iloc[idx][title_col]
                if item_id not in self.recommended_items[domain]:
                    item = df.iloc[idx].copy()
                    item['similarity_score'] = similarities[idx]  # Add similarity score
                    unique_recs.append(item)
                    self.recommended_items[domain].add(item_id)
                if len(unique_recs) >= n_recommendations:
                    break
        
        return pd.DataFrame(unique_recs).reset_index(drop=True).head(n_recommendations)
    
    def _format_recommendations(self, recs, domain, is_similar=False):
        """Format recommendations based on domain"""
        if domain == 'movies' or domain == 'tv_shows':
            return self._format_movie_tv_recommendations(recs, domain, is_similar)
        elif domain == 'music':
            return self._format_music_recommendations(recs, is_similar)
        elif domain == 'books':
            return self._format_book_recommendations(recs, is_similar)
        elif domain == 'food':
            return self._format_food_recommendations(recs, is_similar)
    
    def _format_movie_tv_recommendations(self, recs, domain, is_similar=False):
        """Format movie or TV show recommendations"""
        response = f"Here are some {domain} recommendations for you:\n\n"
        for i, row in recs.iterrows():
            response += f"**{row['title']}** ({row['genre']}) - Rating: {row['rating']}, Mood: {row['mood']}\n"
            response += f"Description: {row['description'][:100]}...\n\n"
        return response
    
    def _format_music_recommendations(self, recs, is_similar=False):
        """Format music recommendations"""
        response = "Here are some music recommendations for you:\n\n"
        for i, row in recs.iterrows():
            response += f"**{row['title']}** by {row['artist']} ({row['genre']}) - Mood: {row['mood']}\n"
            if pd.notna(row['lyrics']) and len(str(row['lyrics'])) > 0:
                response += f"Lyrics excerpt: {str(row['lyrics'])[:50]}...\n\n"
            else:
                response += "\n"
        return response
    
    def _format_book_recommendations(self, recs, is_similar=False):
        """Format book recommendations"""
        response = "Here are some book recommendations for you:\n\n"
        for i, row in recs.iterrows():
            response += f"**{row['title']}** by {row['author']} ({row['genre']}) - Rating: {row['average_rating']}, Mood: {row['mood']}\n"
            response += f"Description: {row['description'][:100]}...\n\n"
        return response
    
    def _format_food_recommendations(self, recs, is_similar=False):
        """Format food recommendations with detailed recipe information"""
        response = "Here are some recipe recommendations for you:\n\n"
        for i, row in recs.iterrows():
            response += f"**{row['name']}** ({row['cuisine_type']}) - Rating: {row['rating']}, Mood: {row['mood']}\n"
            response += f"Ingredients: {row['ingredients']}\n"
            response += f"Preparation: {row['description']}\n"
            
            # Add cooking time and difficulty
            cooking_time = row.get('cooking_time', 'N/A')
            difficulty = row.get('difficulty_level', 'N/A')
            response += f"Cooking time: {cooking_time} minutes, Difficulty: {difficulty}\n\n"
        return response

# Initialize the recommender system
@st.cache_resource
def initialize_recommender():
    movies_df, books_df, food_df, music_df, tv_shows_df = load_data()
    if movies_df is not None:
        return AdvancedRecommender(movies_df, books_df, food_df, music_df, tv_shows_df)
    else:
        return None

# Main app
def main():
    st.markdown(
    """
    <div style="text-align: left;">
        <h1 class='main-header' style="font-size: 36px; margin-bottom: 5px;">
            <span style="font-size:24px;">🎬 📚 🍔 📺 🎵</span>
            Get Recommended :)
        </h1>
        <p style="font-size:12px; color:gray; margin-top:-5px;">
            developed by sai meghana boyapati
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

    st.write("Ask for recommendations across movies, TV shows, music, books, and food!")
    
    # Initialize the recommender with a loading spinner
    with st.spinner("Initializing recommendation system..."):
        recommender = initialize_recommender()
    
    if recommender is None:
        st.error("Failed to initialize the recommender system. Please check your data files.")
        return
    
    # Display example prompts as buttons
    st.sidebar.header("About")
    selected_prompt = None
    # Example prompts
    # Sidebar: Example prompts
    st.sidebar.markdown(
        """
        <p style="font-size:16px;">
            This LLM is an experimental recommendation assistant that can suggest movies, music, books, TV shows, and recipes based on your prompts. You can ask specific, creative, or broad queries, and it will generate tailored suggestions for you.
        </p>

        <h4>Example prompts:</h4>

        <ul>
            <li>Suggest movies with a slow-burn romance</li>
            <li>Recommend animated series for adults</li>
            <li>Share nostalgic 2000s hits</li>
            <li>Recommend books with poetic writing styles</li>
            <li>What are some easy vegetarian dishes?</li>
        </ul>

        <p><em>Note: This model was trained on sample data, so answers are meant as inspiration and may not always be fully accurate.</em></p>
        """,
        unsafe_allow_html=True
    )

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input area
    if prompt := st.chat_input("What would you like recommendations for?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get recommendation
        with st.chat_message("assistant"):
            response = recommender.process_query(prompt)
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
 


if __name__ == "__main__":
    main()
    st.markdown(
        """
        <script>
            var input = window.parent.document.querySelector('input[type="text"]');
            if (input) {
                input.scrollIntoView(false);
            }
        </script>
        """,
        unsafe_allow_html=True
    )