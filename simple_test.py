#!/usr/bin/env python3
"""
Simple test for the enhanced recommendation system
"""

print("Starting test...")

try:
    from recommendation_app import OptimizedMultiDomainRecommendationSystem
    print("✅ Import successful")
    
    # Initialize system
    rec_system = OptimizedMultiDomainRecommendationSystem(data_path='./')
    print("✅ System initialized")
    
    # Test query parsing
    test_query = "Taylor Swift songs"
    params = rec_system.parse_query(test_query)
    print(f"✅ Query parsed: {params}")
    
    # Load data
    print("📊 Loading data...")
    if rec_system.load_preprocessed_data():
        print("✅ Data loaded successfully!")
        
        # Test recommendations
        recommendations = rec_system.get_recommendations(test_query)
        print(f"✅ Got {len(recommendations)} recommendations")
        
        if recommendations:
            print("Sample recommendation:")
            rec = recommendations[0]
            print(f"  Title: {rec.get('title', 'N/A')}")
            print(f"  Artist: {rec.get('artist', 'N/A')}")
            print(f"  Genre: {rec.get('genre', 'N/A')}")
            print(f"  Rating: {rec.get('rating', 'N/A')}")
    else:
        print("❌ Failed to load data")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("Test completed!")
