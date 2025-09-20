#!/usr/bin/env python3
"""
Test script for enhanced recommendation system with direct entity search
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from recommendation_app import OptimizedMultiDomainRecommendationSystem

def test_direct_entity_searches():
    """Test the enhanced system with direct entity searches"""
    print("🎯 Testing Enhanced Recommendation System")
    print("=" * 50)
    
    # Initialize system
    rec_system = OptimizedMultiDomainRecommendationSystem(data_path='./')
    
    # Load data
    print("📊 Loading data...")
    if not rec_system.load_preprocessed_data():
        print("❌ Failed to load data")
        return
    
    print("✅ Data loaded successfully!")
    
    # Test queries for direct entity searches
    test_queries = [
        # Direct artist searches
        "Taylor Swift songs",
        "Coldplay music",
        "Imagine Dragons tracks",
        "Ariana Grande albums",
        
        # Direct actor searches
        "Leonardo DiCaprio movies",
        "Brad Pitt films",
        "Jennifer Lawrence shows",
        "Tom Hanks movies",
        
        # Direct director searches
        "Christopher Nolan movies",
        "Steven Spielberg films",
        "Quentin Tarantino movies",
        "Martin Scorsese films",
        
        # Direct author searches
        "J.K. Rowling books",
        "Agatha Christie novels",
        "Ernest Hemingway books",
        "Jane Austen novels",
        
        # Mixed queries
        "Romantic movies with Leonardo DiCaprio",
        "Action movies by Christopher Nolan",
        "Taylor Swift songs for workout",
        "J.K. Rowling fantasy books",
        
        # Complex queries
        "Movies starring Brad Pitt from 2000s",
        "Taylor Swift songs with high ratings",
        "Books by Agatha Christie with mystery genre",
        "Christopher Nolan movies with high ratings"
    ]
    
    print("\n🔍 Testing Direct Entity Searches:")
    print("=" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i:2d}. Query: '{query}'")
        print("-" * 40)
        
        try:
            # Parse query to see what entities are detected
            params = rec_system.parse_query(query)
            print(f"    Parsed: {params}")
            
            # Get recommendations
            recommendations = rec_system.get_recommendations(query)
            
            if recommendations:
                print(f"    ✅ Found {len(recommendations)} recommendations:")
                for j, rec in enumerate(recommendations[:3], 1):  # Show top 3
                    title = rec.get('title', rec.get('name', 'Unknown'))
                    artist = rec.get('artist', '')
                    director = rec.get('director', '')
                    author = rec.get('author', '')
                    rating = rec.get('rating', rec.get('average_rating', 'N/A'))
                    genre = rec.get('genre', 'N/A')
                    
                    print(f"       {j}. {title}")
                    if artist:
                        print(f"          Artist: {artist}")
                    if director:
                        print(f"          Director: {director}")
                    if author:
                        print(f"          Author: {author}")
                    print(f"          Genre: {genre}, Rating: {rating}")
            else:
                print("    ❌ No recommendations found")
                
        except Exception as e:
            print(f"    ❌ Error: {str(e)}")
    
    print("\n✅ Testing completed!")

if __name__ == "__main__":
    test_direct_entity_searches()
