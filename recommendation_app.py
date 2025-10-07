import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import os
import re
from difflib import get_close_matches

# Set page config FIRST (before any other Streamlit commands)
st.set_page_config(
    page_title="Cross-Domain Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Offline LangChain + LlamaIndex imports
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from llama_index.core import Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.huggingface import HuggingFaceLLM
    OFFLINE_AVAILABLE = True
except ImportError:
    OFFLINE_AVAILABLE = False
    st.warning("Offline capabilities not available. Install: pip install langchain-community llama-index-core llama-index-llms-huggingface llama-index-embeddings-huggingface sentence-transformers faiss-cpu")

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

# Load data from GitHub or local directory - optimized with caching
@st.cache_data(ttl=3600, max_entries=1)
def load_data():
    """Load data with improved error handling and fallbacks"""
    # GitHub repository details
    github_user = "saimeghana9"
    github_repo = "Recommendation_LLM" 
    github_branch = "main"
    
    urls = {
        'movies': f"https://raw.githubusercontent.com/{github_user}/{github_repo}/{github_branch}/movies.csv",
        'books': f"https://raw.githubusercontent.com/{github_user}/{github_repo}/{github_branch}/books.csv",
        'food': f"https://raw.githubusercontent.com/{github_user}/{github_repo}/{github_branch}/food.csv",
        'music': f"https://raw.githubusercontent.com/{github_user}/{github_repo}/{github_branch}/music.csv",
        'tv_shows': f"https://raw.githubusercontent.com/{github_user}/{github_repo}/{github_branch}/tv_shows.csv"
    }
    
    # Try GitHub first
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        loaded_data = {}
        domains = list(urls.keys())
        
        for i, domain in enumerate(domains):
            status_text.text(f"Loading {domain} data from GitHub...")
            loaded_data[domain] = pd.read_csv(urls[domain])
            progress_bar.progress((i + 1) * 20)
        
        status_text.text("Data loaded successfully from GitHub!")
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()
        
        return (loaded_data['movies'], loaded_data['books'], loaded_data['food'], 
                loaded_data['music'], loaded_data['tv_shows'])
        
    except Exception as e:
        st.warning(f"Could not load data from GitHub: {e}")
        
        # Try local directory
        local_path = r"C:\Users\saime\Downloads\Rec"
        
        if os.path.exists(local_path):
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                loaded_data = {}
                domains = ['movies', 'books', 'food', 'music', 'tv_shows']
                
                for i, domain in enumerate(domains):
                    status_text.text(f"Loading {domain} data from local...")
                    file_path = os.path.join(local_path, f'{domain}.csv')
                    loaded_data[domain] = pd.read_csv(file_path)
                    progress_bar.progress((i + 1) * 20)
                
                status_text.text("Data loaded successfully from local directory!")
                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()
                
                return (loaded_data['movies'], loaded_data['books'], loaded_data['food'], 
                        loaded_data['music'], loaded_data['tv_shows'])
                
            except Exception as e:
                st.error(f"Error loading local data: {e}")
        
        # Fallback to sample data
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
    
    # Sample books data - ENHANCED WITH POETIC/LITERARY BOOKS
    books_data = {
        'title': ['To Kill a Mockingbird', '1984', 'The Great Gatsby', 
                 'The Night Circus', 'The Ocean at the End of the Lane', 'The English Patient',
                 'Circe', 'The Song of Achilles', 'All the Light We Cannot See',
                 'The Shadow of the Wind', 'The Book Thief', 'The God of Small Things'],
        'author': ['Harper Lee', 'George Orwell', 'F. Scott Fitzgerald', 
                  'Erin Morgenstern', 'Neil Gaiman', 'Michael Ondaatje',
                  'Madeline Miller', 'Madeline Miller', 'Anthony Doerr',
                  'Carlos Ruiz Zafón', 'Markus Zusak', 'Arundhati Roy'],
        'genre': ['Fiction', 'Dystopian', 'Fiction', 'Fantasy', 'Fantasy', 'Historical Fiction',
                 'Historical Fiction', 'Historical Fiction', 'Historical Fiction',
                 'Mystery', 'Historical Fiction', 'Fiction'],
        'mood': ['Thought-provoking', 'Dark', 'Tragic', 'Magical', 'Whimsical', 'Lyrical',
                'Mythical', 'Epic', 'Poetic', 'Atmospheric', 'Heartbreaking', 'Lyrical'],
        'keywords': ['racism justice childhood', 'totalitarianism surveillance rebellion', 
                    'american dream jazz age', 'magical competition romance poetic writing style',
                    'childhood magic memory lyrical prose', 'war love identity poetic writing beautiful prose',
                    'mythology goddess transformation lyrical writing', 'greek mythology love epic poetic',
                    'world war ii blindness radio poetic descriptive', 'barcelona books mystery atmospheric prose',
                    'nazi germany death books lyrical narrative', 'india family love poetic prose writing style'],
        'average_rating': [4.7, 4.6, 4.3, 4.2, 4.1, 4.0, 4.4, 4.5, 4.3, 4.4, 4.5, 4.2],
        'description': [
            'The story of young Scout Finch and her father, a lawyer who defends a black man accused of raping a white woman in the Depression-era South.',
            'A dystopian social science fiction novel that examines the consequences of totalitarianism, mass surveillance, and repressive regimentation.',
            'A story of Jay Gatsby, a self-made millionaire, and his pursuit of Daisy Buchanan, a wealthy young woman whom he loved in his youth.',
            'A magical competition between two illusionists with beautifully poetic prose and atmospheric writing that will enchant readers.',
            'A magical story of childhood and memory with lyrical, poetic writing that captures the wonder of youth through beautiful prose.',
            'A beautifully written novel about love and identity during World War II with poetic, lyrical prose and elegant writing style.',
            'A retelling of Greek mythology from Circe\'s perspective with lush, poetic language and beautiful prose that mesmerizes readers.',
            'A reimagining of the Achilles and Patroclus story with epic, lyrical writing and emotional depth through poetic narrative.',
            'A Pulitzer Prize-winning novel with poetic descriptions and beautiful prose about a blind French girl and a German boy during WWII.',
            'A novel about a mysterious book with beautiful, atmospheric writing and poetic descriptions of Barcelona that captivate readers.',
            'A story narrated by Death during WWII with unique, poetic language and lyrical prose that creates a hauntingly beautiful narrative.',
            'A novel set in India with poetic, lyrical prose that explores family dynamics and love through beautiful writing style.'
        ]
    }
    
    # Sample food data - ENHANCED WITH VEGETARIAN OPTIONS
    food_data = {
        'name': ['Spaghetti Carbonara', 'Vegetable Stir Fry', 'Greek Salad', 
                'Avocado Toast', 'Vegetable Curry', 'Mushroom Risotto',
                'Vegetable Lasagna', 'Black Bean Burgers', 'Quinoa Bowl'],
        'cuisine_type': ['Italian', 'Asian', 'Greek', 'International', 'Indian', 'Italian',
                        'Italian', 'American', 'International'],
        'mood': ['Comforting', 'Healthy', 'Refreshing', 'Fresh', 'Spicy', 'Creamy',
                'Satisfying', 'Hearty', 'Nutritious'],
        'keywords': ['pasta bacon egg cheese', 'vegetables quick healthy easy vegetarian', 'cucumber tomato feta fresh',
                    'avocado bread simple easy vegetarian', 'vegetables spices coconut milk easy vegetarian', 'rice mushrooms creamy vegetarian',
                    'pasta vegetables cheese easy vegetarian', 'beans burger patty easy vegetarian', 'grains vegetables healthy easy vegetarian'],
        'rating': [4.8, 4.5, 4.3, 4.0, 4.4, 4.2, 4.6, 4.1, 4.3],
        'ingredients': ['Spaghetti, eggs, cheese, pancetta, black pepper', 
                       'Mixed vegetables, soy sauce, garlic, ginger, oil, tofu',
                       'Cucumber, tomato, red onion, feta cheese, olives, olive oil',
                       'Bread, avocado, salt, pepper, olive oil, cherry tomatoes',
                       'Mixed vegetables, curry spices, coconut milk, rice',
                       'Arborio rice, mushrooms, vegetable broth, white wine, parmesan',
                       'Lasagna noodles, mixed vegetables, tomato sauce, cheese',
                       'Black beans, breadcrumbs, spices, burger buns, toppings',
                       'Quinoa, mixed vegetables, chickpeas, dressing, seeds'],
        'description': [
            'A classic Italian pasta dish with a creamy egg-based sauce, pancetta, and cheese.',
            'A quick and healthy dish with fresh vegetables stir-fried with Asian flavors. Easy to make and completely vegetarian.',
            'A refreshing salad with Mediterranean ingredients and a tangy dressing. Simple and vegetarian.',
            'Simple yet delicious toast topped with mashed avocado and seasonings. Quick and easy vegetarian option.',
            'A flavorful curry with mixed vegetables in spiced coconut milk. Easy vegetarian dish ready in 30 minutes.',
            'Creamy Italian rice dish with mushrooms and parmesan cheese. Rich and satisfying vegetarian meal.',
            'Layered pasta with vegetables and cheese in tomato sauce. Comforting and easy vegetarian dinner.',
            'Hearty burger patties made from black beans and spices. Quick and easy vegetarian alternative.',
            'Nutritious bowl with quinoa, fresh vegetables, and protein. Healthy and easy vegetarian meal.'
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
        
        # Offline components (initialized lazily)
        self.rag_embeddings = None
        self.domain_embeddings = {}
        self.domain_vector_stores = {}
        self.domain_retrievers = {}
        self.general_vector_store = None
        self.general_retriever = None
        
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
        """Prepare data for each domain with combined text features - optimized"""
        # Fast string concatenation using vectorized operations
        def create_combined_text(df, base_cols, extra_cols=None):
            if extra_cols is None:
                extra_cols = []
            
            # Combine base columns
            combined = df[base_cols].astype(str).agg(' '.join, axis=1)
            
            # Add extra columns if they exist
            for col in extra_cols:
                if col in df.columns:
                    combined += ' ' + df[col].astype(str)
            
            return combined.fillna('')
        
        # Movies - optimized
        self.movies_df['combined_text'] = create_combined_text(
            self.movies_df, 
            ['title', 'genre', 'mood', 'keywords'],
            ['director', 'cast', 'setting', 'time_period']
        )
        
        # Books - optimized with enhanced features for literary matching
        self.books_df['combined_text'] = create_combined_text(
            self.books_df,
            ['title', 'author', 'genre', 'mood', 'keywords', 'description'],
            ['setting', 'time_period']
        )
        
        # Food - optimized
        self.food_df['combined_text'] = create_combined_text(
            self.food_df,
            ['name', 'cuisine_type', 'mood', 'keywords', 'ingredients', 'description'],
            ['meal_type', 'dish_type', 'tags', 'category']
        )
        
        # Music - optimized
        self.music_df['combined_text'] = create_combined_text(
            self.music_df,
            ['title', 'artist', 'genre', 'mood', 'keywords'],
            ['album', 'year', 'instrumentation', 'lyrics']
        )
        
        # TV Shows - optimized
        self.tv_shows_df['combined_text'] = create_combined_text(
            self.tv_shows_df,
            ['title', 'genre', 'mood', 'keywords'],
            ['creator', 'setting', 'time_period']
        )
    
    def train_tfidf_models(self):
        """Train TF-IDF models for each domain with better memory management"""
        self.tfidf_vectorizers = {}
        self.tfidf_matrices = {}
        
        domains = {
            'movies': self.movies_df,
            'books': self.books_df, 
            'food': self.food_df,
            'music': self.music_df,
            'tv_shows': self.tv_shows_df
        }
        
        # More efficient TF-IDF parameters
        vectorizer_params = {
            'max_features': 2000,  # Increased for better matching
            'stop_words': 'english',
            'ngram_range': (1, 3),  # Include trigrams for better phrase matching
            'min_df': 1,
            'max_df': 0.95,
            'dtype': np.float32  # Use smaller data type
        }
        
        for domain, df in domains.items():
            if len(df) > 0:
                try:
                    vectorizer = TfidfVectorizer(**vectorizer_params)
                    tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
                    self.tfidf_vectorizers[domain] = vectorizer
                    self.tfidf_matrices[domain] = tfidf_matrix
                except Exception as e:
                    st.warning(f"Failed to train TF-IDF for {domain}: {e}")
    
    def detect_domain_improved(self, query: str):
        """IMPROVED domain detection with better logic for ambiguous queries"""
        query_lower = query.lower().strip()
        
        # First, check for explicit domain mentions
        domain_indicators = {
            'books': ['book', 'novel', 'read', 'author', 'literary', 'poetic', 'writing', 'prose', 'story'],
            'food': ['food', 'recipe', 'dish', 'cook', 'meal', 'eat', 'dinner', 'lunch', 'breakfast', 
                    'vegetarian', 'vegan', 'ingredient', 'cuisine', 'cooking'],
            'movies': ['movie', 'film', 'cinema', 'watch', 'actor', 'director', 'oscar'],
            'tv_shows': ['tv', 'television', 'show', 'series', 'episode', 'season', 'netflix', 'hulu'],
            'music': ['music', 'song', 'track', 'album', 'artist', 'band', 'listen', 'playlist']
        }
        
        # Count domain indicators
        domain_scores = {domain: 0 for domain in domain_indicators}
        
        for domain, indicators in domain_indicators.items():
            for indicator in indicators:
                if indicator in query_lower:
                    domain_scores[domain] += 1
        
        # Get the domain with highest score
        best_domain = max(domain_scores, key=domain_scores.get)
        
        # If we have a clear winner (at least 2 points more than others), return it
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_domains) > 1:
            if sorted_domains[0][1] >= sorted_domains[1][1] + 2:
                return sorted_domains[0][0]
        
        # For specific query patterns, use direct mapping
        query_patterns = {
            'books': [
                r'.*book.*poetic.*writing.*',
                r'.*poetic.*writing.*style.*',
                r'.*literary.*style.*',
                r'.*beautiful.*prose.*',
                r'.*lyrical.*writing.*'
            ],
            'food': [
                r'.*easy.*vegetarian.*',
                r'.*simple.*vegetarian.*',
                r'.*vegetarian.*dish.*',
                r'.*easy.*recipe.*',
                r'.*simple.*meal.*'
            ],
            'movies': [
                r'.*iconic.*classic.*',
                r'.*classic.*movie.*',
                r'.*must.*watch.*film.*',
                r'.*great.*movie.*'
            ]
        }
        
        for domain, patterns in query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return domain
        
        # For comfort/general recommendations, distribute across domains
        comfort_terms = ['comfort', 'cozy', 'relaxing', 'feel-good', 'great', 'good', 'recommendation']
        if any(term in query_lower for term in comfort_terms):
            # Return the domain with highest score, or default to movies
            if domain_scores[best_domain] > 0:
                return best_domain
            return 'movies'
        
        # If no clear domain, return None (will use RAG)
        if domain_scores[best_domain] == 0:
            return None
        
        return best_domain
    
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
                          'food', 'recipe', 'tv', 'show', 'shows', 'romcom', 'romantic', 'comedy',
                          'poetic', 'writing', 'style', 'literary', 'vegetarian', 'dish', 'easy']  # Added more terms
            
            close_matches = get_close_matches(word, domain_terms, n=1, cutoff=0.7)
            if close_matches:
                corrected_words.append(close_matches[0])
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    def detect_domain(self, query: str):
        """Enhanced domain detection with spelling correction and improved logic"""
        # Use the improved detection as primary
        domain = self.detect_domain_improved(query)
        if domain:
            return domain
        
        # Fallback to original detection
        corrected_query = self.correct_spelling(query)
        query_lower = corrected_query.lower()
        
        # Single word domain mapping
        single_word_domains = {
            'movies': ['movie', 'film', 'cinema', 'romcom', 'thriller', 'comedy', 'drama', 'action', 'classic'],
            'tv_shows': ['tv', 'show', 'series', 'sitcom', 'kdrama'],
            'music': ['music', 'song', 'track', 'album', 'jazz', 'rock', 'pop'],
            'books': ['book', 'novel', 'read', 'fiction', 'fantasy', 'romance', 'poetic', 'literary'],
            'food': ['food', 'recipe', 'dish', 'cooking', 'meal', 'pasta', 'pizza', 'vegetarian']
        }
        
        # Check for single word queries
        if len(query_lower.split()) == 1:
            for domain, words in single_word_domains.items():
                if query_lower in words:
                    return domain
        
        # Comprehensive domain mapping
        domain_keywords = {
            'books': [
                'book', 'novel', 'read', 'author', 'page', 'chapter', 'story', 'literary',
                'poetic', 'writing', 'prose', 'literature', 'fiction', 'nonfiction', 'bestseller',
                'classic', 'contemporary', 'modern', 'historical', 'fantasy', 'romance', 'mystery',
                'thriller', 'biography', 'memoir', 'autobiography', 'poetry', 'essay', 'short story',
                'publisher', 'edition', 'hardcover', 'paperback', 'ebook', 'audiobook', 'bookstore',
                'library', 'reading list', 'book club', 'must read', 'highly recommended'
            ],
            'food': [
                'food', 'recipe', 'dish', 'meal', 'cooking', 'cook', 'eat', 'dining', 'cuisine',
                'ingredient', 'vegetarian', 'vegan', 'healthy', 'easy', 'simple', 'quick', 'fast',
                'delicious', 'tasty', 'flavorful', 'nutritious', 'homemade', 'restaurant', 'chef',
                'kitchen', 'cookbook', 'menu', 'appetizer', 'main course', 'entree', 'side dish',
                'dessert', 'snack', 'breakfast', 'lunch', 'dinner', 'supper', 'brunch', 'buffet',
                'ingredients', 'instructions', 'preparation', 'cook time', 'servings', 'calories'
            ],
            'movies': [
                'movie', 'film', 'cinema', 'watch', 'actor', 'actress', 'director', 'producer',
                'screenplay', 'script', 'scene', 'sequel', 'prequel', 'remake', 'adaptation',
                'oscar', 'award', 'nomination', 'blockbuster', 'indie', 'independent', 'hollywood',
                'bollywood', 'documentary', 'animation', 'animated', 'live action', 'thriller',
                'comedy', 'drama', 'action', 'adventure', 'romance', 'horror', 'sci-fi', 'fantasy',
                'mystery', 'crime', 'western', 'historical', 'biographical', 'musical', 'classic',
                'contemporary', 'modern', 'cult', 'iconic', 'must see', 'highly rated', 'acclaimed'
            ]
        }
        
        # Score each domain based on keyword matches
        domain_scores = {domain: 0 for domain in domain_keywords}
        
        for domain, keywords in domain_keywords.items():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', query_lower):
                    domain_scores[domain] += 2
                elif keyword in query_lower:
                    domain_scores[domain] += 1
        
        # Find the domain with the highest score
        best_domain = max(domain_scores, key=domain_scores.get)
        
        # Return domain if it has reasonable score
        if domain_scores[best_domain] >= 2:
            return best_domain
        
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
                'romcom': ['romantic comedy', 'romance', 'comedy', 'love story'],
                'iconic classics': ['classic', 'timeless', 'iconic', 'must-see', 'essential']
            },
            'food': {
                'pasta': ['noodles', 'spaghetti', 'macaroni', 'penne', 'fettuccine', 'linguine'],
                'recipes': ['dish', 'meal', 'cooking', 'preparation'],
                'simple': ['easy', 'quick', 'basic', 'minimal', 'straightforward'],
                'impressive': ['elegant', 'fancy', 'gourmet', 'sophisticated', 'restaurant-quality'],
                'vegetarian': ['meatless', 'plant-based', 'veggie', 'meat-free', 'vegetable'],
                'easy vegetarian': ['simple', 'quick', 'fast', 'straightforward', 'uncomplicated']
            },
            'music': {
                'love': ['romantic', 'heartfelt', 'emotional', 'passionate'],
                'relaxing': ['calming', 'soothing', 'peaceful', 'tranquil'],
                'energetic': ['upbeat', 'lively', 'dynamic', 'vibrant']
            },
            'books': {
                # Enhanced book query enhancers for literary and poetic queries
                'love': ['romance', 'relationship', 'heartfelt', 'emotional'],
                'thriller': ['suspense', 'mystery', 'crime', 'intrigue'],
                'poetic': ['lyrical', 'beautiful prose', 'elegant writing', 'literary', 'descriptive language'],
                'writing style': ['prose', 'narrative style', 'literary style', 'language', 'voice'],
                'literary': ['poetic', 'lyrical', 'well-written', 'beautiful language', 'prose'],
                'beautiful writing': ['lyrical prose', 'elegant language', 'descriptive', 'atmospheric'],
                'prose': ['writing style', 'narrative', 'language', 'literary style']
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
    
    def get_direct_recommendations(self, domain: str, query: str, n_recommendations: int = 3):
        """Get direct recommendations bypassing domain detection issues"""
        try:
            if domain not in self.tfidf_vectorizers:
                return pd.DataFrame()
            
            vectorizer = self.tfidf_vectorizers[domain]
            tfidf_matrix = self.tfidf_matrices[domain]
            
            query_vec = vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
            
            df = getattr(self, f"{domain}_df")
            title_col = 'title' if domain != 'food' else 'name'
            
            # Get top indices
            top_indices = similarities.argsort()[::-1][:n_recommendations*2]
            
            candidates = []
            for idx in top_indices:
                if similarities[idx] > 0.01:  # Very low threshold
                    item_id = df.iloc[idx][title_col]
                    if item_id not in self.recommended_items[domain]:
                        item = df.iloc[idx].copy()
                        item['similarity_score'] = similarities[idx]
                        candidates.append(item)
                        self.recommended_items[domain].add(item_id)
                    if len(candidates) >= n_recommendations:
                        break
            
            if candidates:
                return pd.DataFrame(candidates).head(n_recommendations)
            
            # If no good matches, return top rated items
            rating_col = 'average_rating' if 'average_rating' in df.columns else 'rating'
            if rating_col in df.columns:
                return df.nlargest(n_recommendations, rating_col)
            
            return df.head(n_recommendations)
            
        except Exception as e:
            st.warning(f"Error in direct recommendations for {domain}: {e}")
            return pd.DataFrame()
    
    def process_query(self, query: str):
        """Process a user query and return recommendations - with improved domain detection"""
        # First try improved domain detection
        domain = self.detect_domain(query)
        
        if not domain:
            # For comfort/general queries, try multiple domains
            if any(term in query.lower() for term in ['comfort', 'cozy', 'great', 'good', 'recommendation']):
                return self.handle_general_recommendations(query)
            
            # Try RAG for cross-domain queries
            if OFFLINE_AVAILABLE and hasattr(self, 'general_retriever') and self.general_retriever is not None:
                try:
                    return self.conversational_query(query)
                except:
                    pass
            
            return "I can help with recommendations for movies, TV shows, music, books, and food. Please specify what you're looking for!"
        
        # Enhanced query processing
        enhanced_query = self.enhance_query(query, domain)
        
        # Get recommendations
        recs = self.get_direct_recommendations(domain, enhanced_query, 5)
        
        # Special handling for specific query types
        if domain == 'books' and any(term in query.lower() for term in ['poetic', 'writing style', 'literary', 'prose']):
            poetic_books = self.books_df[
                self.books_df['mood'].str.contains('poetic|lyrical|literary', case=False, na=False) |
                self.books_df['keywords'].str.contains('poetic|lyrical|writing style|beautiful prose', case=False, na=False)
            ]
            if len(poetic_books) > 0:
                poetic_recs = poetic_books.head(3).copy()
                poetic_recs['similarity_score'] = 0.9
                all_recs = pd.concat([recs, poetic_recs]).drop_duplicates(subset=['title']).head(3)
                if len(all_recs) > 0:
                    return self._format_recommendations(all_recs, domain, False)
        
        elif domain == 'food' and any(term in query.lower() for term in ['vegetarian', 'vegan']):
            veg_food = self.food_df[
                self.food_df['keywords'].str.contains('vegetarian|vegan', case=False, na=False)
            ]
            if len(veg_food) > 0:
                veg_recs = veg_food.head(3).copy()
                veg_recs['similarity_score'] = 0.9
                all_recs = pd.concat([recs, veg_recs]).drop_duplicates(subset=['name']).head(3)
                if len(all_recs) > 0:
                    return self._format_recommendations(all_recs, domain, False)
        
        # Final fallback
        if len(recs) == 0:
            # Return top rated items from the domain
            df = getattr(self, f"{domain}_df")
            rating_col = 'average_rating' if 'average_rating' in df.columns else 'rating'
            if rating_col in df.columns:
                fallback_recs = df.nlargest(3, rating_col)
            else:
                fallback_recs = df.head(3)
            
            if len(fallback_recs) > 0:
                return f"Here are some highly-rated {domain} you might enjoy:\n\n" + \
                       self._format_recommendations(fallback_recs, domain, True)
            else:
                return f"Sorry, I couldn't find specific {domain} recommendations for '{query}'. Try a different query!"
        
        return self._format_recommendations(recs.head(3), domain, False)
    
    def handle_general_recommendations(self, query: str):
        """Handle general/comfort recommendations across domains"""
        responses = []
        domains_to_try = ['movies', 'books', 'food', 'music', 'tv_shows']
        
        for domain in domains_to_try:
            recs = self.get_direct_recommendations(domain, query, 2)
            if len(recs) > 0:
                responses.append(f"**{domain.title()}:**\n{self._format_recommendations(recs, domain, False)}")
        
        if responses:
            return "Here are some comfort recommendations across different categories:\n\n" + "\n\n".join(responses)
        else:
            return "I can help with recommendations for movies, TV shows, music, books, and food. Please specify what you're looking for!"
    
    def _get_cached_domain(self, query: str):
        """Cached domain detection for faster processing"""
        if not hasattr(self, '_domain_cache'):
            self._domain_cache = {}
        
        query_key = query.lower().strip()
        if query_key in self._domain_cache:
            return self._domain_cache[query_key]
        
        domain = self.detect_domain(query)
        self._domain_cache[query_key] = domain
        return domain
    
    def _find_artists_fast(self, query: str):
        """Fast artist detection using pre-computed set"""
        query_lower = query.lower()
        found_artists = []
        for artist in self.music_artists:
            if artist in query_lower:
                found_artists.append(artist)
        return found_artists
    
    def get_recommendations(self, domain: str, query: str, n_recommendations: int = 3):
        """Get recommendations using TF-IDF and cosine similarity"""
        return self.get_direct_recommendations(domain, query, n_recommendations)
    
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
        """Format book recommendations with enhanced information"""
        response = "Here are some book recommendations for you:\n\n"
        for i, row in recs.iterrows():
            response += f"**{row['title']}** by {row['author']}\n"
            response += f"- Genre: {row['genre']}, Mood: {row['mood']}\n"
            if 'average_rating' in row and pd.notna(row['average_rating']):
                response += f"- Rating: {row['average_rating']}/5\n"
            response += f"- Description: {row['description'][:150]}...\n\n"
        return response
    
    def _format_food_recommendations(self, recs, is_similar=False):
        """Format food recommendations with detailed recipe information"""
        response = "Here are some recipe recommendations for you:\n\n"
        for i, row in recs.iterrows():
            response += f"**{row['name']}** ({row['cuisine_type']}) - Rating: {row['rating']}, Mood: {row['mood']}\n"
            response += f"Ingredients: {row['ingredients'][:100]}...\n"
            response += f"Preparation: {row['description'][:100]}...\n\n"
        return response

    def _ensure_offline_components(self):
        """Initialize offline components if available - optimized"""
        if not OFFLINE_AVAILABLE:
            return False
            
        if self.rag_embeddings is None:
            try:
                # Use caching for embeddings with faster model
                if not hasattr(self, '_embeddings_loaded'):
                    with st.spinner("Loading embeddings (one-time setup)..."):
                        # Use a smaller, faster model for better performance
                        self.rag_embeddings = HuggingFaceEmbeddings(
                            model_name="sentence-transformers/all-MiniLM-L6-v2",
                            model_kwargs={'device': 'cpu'},  # Force CPU for consistency
                            encode_kwargs={'normalize_embeddings': True}
                        )
                        self._embeddings_loaded = True
                return True
            except Exception as e:
                st.warning(f"Failed to initialize offline embeddings: {e}")
                return False
        return True
    
    def _build_sentencebert_index_for_domain(self, domain: str):
        """Build a FAISS store for a domain using sentence-BERT"""
        if not self._ensure_offline_components():
            return
            
        if domain in self.domain_embeddings:
            return

        if domain == 'movies':
            df = self.movies_df.copy()
            texts = (df['title'] + ' | ' + df['genre'] + ' | ' + df['mood'] + ' | ' + df['description']).fillna('').tolist()
            title_col = 'title'
        elif domain == 'tv_shows':
            df = self.tv_shows_df.copy()
            texts = (df['title'] + ' | ' + df['genre'] + ' | ' + df['mood'] + ' | ' + df['description']).fillna('').tolist()
            title_col = 'title'
        elif domain == 'music':
            df = self.music_df.copy()
            texts = (df['title'] + ' | ' + df['artist'] + ' | ' + df['genre'] + ' | ' + df.get('lyrics','')).fillna('').tolist()
            title_col = 'title'
        elif domain == 'books':
            df = self.books_df.copy()
            texts = (df['title'] + ' | ' + df['author'] + ' | ' + df['genre'] + ' | ' + df['mood'] + ' | ' + df['description']).fillna('').tolist()
            title_col = 'title'
        elif domain == 'food':
            df = self.food_df.copy()
            texts = (df['name'] + ' | ' + df['cuisine_type'] + ' | ' + df['mood'] + ' | ' + df['ingredients'] + ' | ' + df.get('description','')).fillna('').tolist()
            title_col = 'name'
        else:
            return

        try:
            store = FAISS.from_texts(texts, embedding=self.rag_embeddings)
            self.domain_embeddings[domain] = {
                "vectorstore": store,
                "df": df,
                "title_col": title_col,
                "texts": texts
            }
        except Exception as e:
            st.warning(f"Failed to build sentence-BERT index for {domain}: {e}")
    
    def sentence_bert_recommendations(self, domain: str, query: str, k: int = 5) -> pd.DataFrame:
        """Get recommendations using sentence-BERT + FAISS"""
        if not self._ensure_offline_components():
            return pd.DataFrame()
            
        if domain not in self.domain_embeddings:
            self._build_sentencebert_index_for_domain(domain)
            
        if domain not in self.domain_embeddings:
            return pd.DataFrame()

        store = self.domain_embeddings[domain]["vectorstore"]
        df = self.domain_embeddings[domain]["df"]
        texts = self.domain_embeddings[domain]["texts"]
        title_col = self.domain_embeddings[domain]["title_col"]

        try:
            docs = store.similarity_search(query or "recommend", k=k*2)
            hits = []
            for d in docs:
                page = d.page_content
                try:
                    idx = texts.index(page)
                    hits.append(df.iloc[idx])
                except ValueError:
                    continue

            out = pd.DataFrame(hits).drop_duplicates(subset=[title_col]).head(k).reset_index(drop=True)
            return out
        except Exception as e:
            st.warning(f"Error in sentence-BERT recommendations: {e}")
            return pd.DataFrame()
    
    def setup_rag_system(self):
        """Set up RAG system for conversational queries with better error handling"""
        if not self._ensure_offline_components():
            st.warning("RAG system unavailable - offline components not loaded")
            return

        # Check if already set up
        if hasattr(self, 'general_retriever') and self.general_retriever is not None:
            return

        try:
            with st.spinner("Building RAG vector stores (one-time setup)..."):
                self.domain_vector_stores = {}

                # Process domains with error handling
                domains_to_process = [
                    ('movies', self.movies_df, lambda r: f"Movie: {r.title}. Description: {r.description}. Genre: {r.genre}. Mood: {r.mood}"),
                    ('tv_shows', self.tv_shows_df, lambda r: f"TV Show: {r.title}. Description: {r.description}. Genre: {r.genre}. Mood: {r.mood}"),
                    ('music', self.music_df, lambda r: f"Music: {r.title}. Artist: {r.artist}. Genre: {r.genre}. Mood: {r.mood}"),
                    ('books', self.books_df, lambda r: f"Book: {r.title}. Author: {r.author}. Genre: {r.genre}. Mood: {r.mood}. Description: {r.description}"),
                    ('food', self.food_df, lambda r: f"Food: {r.name}. Cuisine: {r.cuisine_type}. Mood: {r.mood}. Ingredients: {r.ingredients}")
                ]

                all_data = []
                for domain, df, formatter in domains_to_process:
                    try:
                        if len(df) > 0:
                            domain_data = [formatter(row) for _, row in df.iterrows()]
                            self.domain_vector_stores[domain] = FAISS.from_texts(domain_data, self.rag_embeddings)
                            all_data.extend(domain_data)
                    except Exception as e:
                        st.warning(f"Failed to process {domain} for RAG: {e}")

                # Create general retriever
                if all_data:
                    self.general_vector_store = FAISS.from_texts(all_data, self.rag_embeddings)
                    self.domain_retrievers = {d: s.as_retriever(search_kwargs={"k": 3}) 
                                            for d, s in self.domain_vector_stores.items()}
                    self.general_retriever = self.general_vector_store.as_retriever(search_kwargs={"k": 3})
                    
        except Exception as e:
            st.error(f"Failed to setup RAG system: {e}")
    
    def conversational_query(self, query: str):
        """Handle conversational queries using RAG"""
        if not self._ensure_offline_components() or not hasattr(self, 'general_retriever'):
            return "RAG system not available. Please install required packages."
            
        q = query.lower()
        domain = None
        if any(k in q for k in ['movie', 'film', 'cinema']): domain = 'movies'
        elif any(k in q for k in ['tv', 'show', 'series', 'episode']): domain = 'tv_shows'
        elif any(k in q for k in ['music', 'song', 'album', 'track']): domain = 'music'
        elif any(k in q for k in ['book', 'novel', 'read']): domain = 'books'
        elif any(k in q for k in ['food', 'recipe', 'dish', 'cook']): domain = 'food'

        try:
            docs = self.domain_retrievers[domain].get_relevant_documents(query) if (domain and domain in self.domain_retrievers) \
                   else self.general_retriever.get_relevant_documents(query)

            response = "Based on your query, here are some recommendations:\n\n"
            for i, doc in enumerate(docs, start=1):
                content = doc.page_content
                if "Movie:" in content:
                    title = content.split("Movie: ")[1].split(". ")[0]
                    genre = content.split("Genre: ")[1].split(". ")[0] if "Genre: " in content else "Unknown"
                    response += f"{i}. Movie: {title} ({genre})\n"
                elif "TV Show:" in content:
                    title = content.split("TV Show: ")[1].split(". ")[0]
                    genre = content.split("Genre: ")[1].split(". ")[0] if "Genre: " in content else "Unknown"
                    response += f"{i}. TV Show: {title} ({genre})\n"
                elif "Music:" in content:
                    title = content.split("Music: ")[1].split(". ")[0]
                    artist = content.split("Artist: ")[1].split(". ")[0] if "Artist: " in content else "Unknown"
                    response += f"{i}. Music: {title} by {artist}\n"
                elif "Book:" in content:
                    title = content.split("Book: ")[1].split(". ")[0]
                    author = content.split("Author: ")[1].split(". ")[0] if "Author: " in content else "Unknown"
                    response += f"{i}. Book: {title} by {author}\n"
                elif "Food:" in content:
                    name = content.split("Food: ")[1].split(". ")[0]
                    cuisine = content.split("Cuisine: ")[1].split(". ")[0] if "Cuisine: " in content else "Unknown"
                    response += f"{i}. Food: {name} ({cuisine})\n"
                else:
                    response += f"{i}. {content[:200]}...\n"
            return response
        except Exception as e:
            return f"Error in conversational query: {e}"
    
    def process_multiple_queries(self, query: str):
        """Handle multiple prompts in a single query with better parsing"""
        # Enhanced query splitting with context awareness
        separators = [' and ', ' also ', ' plus ', ' & ', ' + ', '\n', ';', ',']
        
        queries = [query]
        for sep in separators:
            if sep in query.lower():
                parts = [part.strip() for part in query.split(sep) if part.strip()]
                if len(parts) > 1:
                    queries = parts
                    break
        
        # Process queries
        if len(queries) == 1:
            return self.process_query(queries[0])
        
        responses = []
        for i, q in enumerate(queries, 1):
            response = self.process_query(q)
            responses.append(f"**Query {i}: {q}**\n{response}")
            
            # Add separator between responses (except for the last one)
            if i < len(queries):
                responses.append("---")
        
        return "\n\n".join(responses)

# Initialize the recommender system with aggressive caching
@st.cache_resource(ttl=3600)  # Cache for 1 hour
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
    
    # Fast initialization with cached data and progress
    if 'recommender' not in st.session_state:
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🚀 Loading recommendation system...")
            progress_bar.progress(20)
            
            st.session_state.recommender = initialize_recommender()
            progress_bar.progress(100)
            status_text.text("You are on!")
            
            # Clear progress indicators after a short delay
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
        except Exception as e:
            st.error(f"Failed to initialize recommender: {e}")
            return
    
    recommender = st.session_state.recommender
    
    if recommender is None:
        st.error("Failed to initialize the recommender system. Please check your data files.")
        return
    
    # Show status (cached)
    if 'status_shown' not in st.session_state:
        if OFFLINE_AVAILABLE:
            st.success("TEST")
        else:
            st.info("ℹ️ Basic mode (TF-IDF only). Install offline packages for enhanced features.")
        st.session_state.status_shown = True
    
    # Initialize RAG system lazily (only when needed)
    if OFFLINE_AVAILABLE and not hasattr(recommender, 'general_retriever'):
        if 'rag_setup' not in st.session_state:
            try:
                with st.spinner("Setting up offline RAG system..."):
                    recommender.setup_rag_system()
                    st.session_state.rag_setup = True
            except Exception as e:
                st.warning(f"RAG setup failed: {e}. Continuing with basic features.")
    
    # Sidebar with information and examples
    st.sidebar.header("About")
    st.sidebar.markdown(
        """
        <p style="font-size:16px;">
            This is an experimental recommendation assistant that can suggest movies, music, books, TV shows, and recipes based on your prompts.You can ask specific, creative, or broad queries, and it will generate tailored suggestions for you.
 

        <h4>Example prompts:</h4>

        <ul>
            <li>Suggest movies with a slow-burn romance</li>
            <li>Recommend animated series for adults</li>
            <li>Share nostalgic 2000s hits</li>
            <li>Recommend books with poetic writing styles</li>
            <li>What are some easy vegetarian dishes?</li>
        </ul>


        <p><em>Note: This model was trained using sample data, so its responses are intended for inspiration and may not always be completely accurate. The system operates entirely offline using local models—no API keys are needed!

        </em></p>
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
        
        # Get recommendation with fast response indicator
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    response = recommender.process_multiple_queries(prompt)
                except Exception as e:
                    response = f"Sorry, I encountered an error: {str(e)}. Please try a different query."
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