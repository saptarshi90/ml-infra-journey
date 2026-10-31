def find_max_average_telemetry(metrics, k):
    # Calculate the sum of the first window
    window_sum = sum(metrics[:k])
    max_sum = window_sum
    
    # Slide the window across the remaining metrics array
    for i in range(k, len(metrics)):
        # Subtract the element leaving the window, add the element entering
        window_sum += metrics[i] - metrics[i - k]
        if window_sum > max_sum:
            max_sum = window_sum
            
    return round(max_sum / k, 2)

# Simulated streaming telemetry data (CPU or RAM usage metrics)
live_stream = [23.5, 45.1, 89.2, 12.4, 56.8, 91.3, 10.5, 44.2]
window_size = 3

result = find_max_average_telemetry(live_stream, window_size)
print(f"🚀 Optimized Stream Processing Success!")
print(f"Maximum average utilization across a {window_size}-second window: {result}%")
