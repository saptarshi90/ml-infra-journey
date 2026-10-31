from collections import deque

class MyCircularQueue:
    """Low-level Circular Queue (Ring Buffer) built with fixed allocation memory."""
    def __init__(self, k: int):
        self.queue = [0] * k
        self.size = k
        self.head = 0
        self.tail = 0
        self.count = 0

    def enQueue(self, value: int) -> bool:
        """Pushes data into the streaming loop using modulo wraps in O(1) time."""
        if self.isFull():
            return False
        self.queue[self.tail] = value
        self.tail = (self.tail + 1) % self.size
        self.count += 1
        return True

    def deQueue(self) -> bool:
        """Pops data from the streaming head using modulo wraps in O(1) time."""
        if self.isEmpty():
            return False
        self.head = (self.head + 1) % self.size
        self.count -= 1
        return True

    def Front(self) -> int:
        return -1 if self.isEmpty() else self.queue[self.head]

    def Rear(self) -> int:
        return -1 if self.isEmpty() else self.queue[(self.tail - 1 + self.size) % self.size]

    def isEmpty(self) -> bool:
        return self.count == 0

    def isFull(self) -> bool:
        return self.count == self.size


class MonotonicTelemetryFilter:
    """Advanced Monotonic Queue to filter maximum metrics out of a live data stream."""
    def __init__(self):
        self.deque = deque()  # Stores indices of elements in strict decreasing value order

    def process_stream(self, nums: list, k: int) -> list:
        """Extracts max values across a sliding window of size k in O(N) overall time."""
        max_metrics = []
        for i, val in enumerate(nums):
            # 1. Prune indices that have fallen completely out of the active window frame
            if self.deque and self.deque[0] < i - k + 1:
                self.deque.popleft()
            
            # 2. Monotonic Maintenance: Evict smaller elements that can never be the maximum
            while self.deque and nums[self.deque[-1]] <= val:
                self.deque.pop()
                
            # 3. Cache current metric element index track
            self.deque.append(i)
            
            # 4. Append maximum value to the results ledger once window is full
            if i >= k - 1:
                max_metrics.append(nums[self.deque[0]])
        return max_metrics

# ==========================================
# SYSTEM ENGINE COMPILATION CHECK
# ==========================================
print("🚀 Verification 1: Booting Circular Ring Buffer (Capacity = 3)")
ring_buffer = MyCircularQueue(3)
print(f"-> Enqueue Task A (10): {ring_buffer.enQueue(10)}")
print(f"-> Enqueue Task B (20): {ring_buffer.enQueue(20)}")
print(f"-> Enqueue Task C (30): {ring_buffer.enQueue(30)}")
print(f"-> Buffer Full Check: {ring_buffer.isFull()} (Target: True)")
print(f"-> Dequeue Task A: {ring_buffer.deQueue()}")
print(f"-> Enqueue Task D (40) [Modulo Loop Wrap]: {ring_buffer.enQueue(40)}")
print(f"-> Active Rear Task Element: {ring_buffer.Rear()} (Target: 40)")

print("\n📡 Verification 2: Running Live Telemetry Max Load Sliding Filter (Window = 3s)")
# Simulated CPU Load percentages scraped every second from cluster pods
cpu_metrics_stream = [40, 10, 20, 80, 50, 60, 30]
filter_engine = MonotonicTelemetryFilter()
output_max_windows = filter_engine.process_stream(cpu_metrics_stream, 3)
print(f"Inbound Data Load Stream: {cpu_metrics_stream}")
print(f"Calculated Max Loads Array:  {output_max_windows}")
print("   * Window [40,10,20] -> Max: 40")
print("   * Window [10,20,80] -> Max: 80")
print("   * Window [20,80,50] -> Max: 80")
print("   * Window [80,50,60] -> Max: 80")
print("   * Window [50,60,30] -> Max: 60")
