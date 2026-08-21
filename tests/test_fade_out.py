import random

def get_rigged_options(total_eps):
    if total_eps < 300_000:
        prob = 1.0
    elif total_eps < 500_000:
        prob = 1.0 - ((total_eps - 300_000) / 200_000)
    else:
        prob = 0.0
        
    if random.random() < prob:
        forced_bid_id = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        return {"forced_bid_id": forced_bid_id}, True
    return None, False

def test_switch():
    test_points = [0, 150_000, 300_000, 350_000, 400_000, 450_000, 500_000, 600_000]
    expected_probs = [1.0, 1.0, 1.0, 0.75, 0.5, 0.25, 0.0, 0.0]
    
    print("Testing Oracle Fade-Out Switch...")
    print("-" * 55)
    
    for eps, expected in zip(test_points, expected_probs):
        rigged_count = 0
        trials = 20000
        for _ in range(trials):
            _, is_rigged = get_rigged_options(eps)
            if is_rigged:
                rigged_count += 1
        
        actual_prob = rigged_count / trials
        print(f"Episodes: {eps:7d} | Expected Rigged: {expected*100:5.1f}% | Actual Rigged: {actual_prob*100:5.1f}%")
        
        assert abs(actual_prob - expected) < 0.02, f"Failed at {eps} eps!"
        
    print("-" * 55)
    print("ALL TESTS PASSED. The Scheduled Sampling math is perfectly smooth.")

if __name__ == '__main__':
    test_switch()
