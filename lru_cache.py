class Node:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # Hash Map linking keys to Node pointers
        
        # Initialize sentinel Dummy Head & Dummy Tail nodes to prevent boundary errors
        self.head = Node(0, 0)
        self.tail = Node(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node):
        """Low-level pointer mutation: Cut a node out from the middle of the chain."""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _add_to_head(self, node: Node):
        """Low-level pointer mutation: Splice a node directly behind the dummy head."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key: int) -> int:
        """Fetch cached data in O(1) time and promote it to Most Recently Used."""
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            self._add_to_head(node)
            return node.value
        return -1

    def put(self, key: int, value: int) -> None:
        """Write data to RAM. Evict the Least Recently Used node if capacity blows out."""
        if key in self.cache:
            self._remove(self.cache[key])
            
        new_node = Node(key, value)
        self._add_to_head(new_node)
        self.cache[key] = new_node
        
        if len(self.cache) > self.capacity:
            # Capacity cap violated! Evict the least recently used node right before the tail
            lru_node = self.tail.prev
            self._remove(lru_node)
            del self.cache[lru_node.key]  # Clear the key registry from the Hash Map

# ==========================================
# SIMULATION VALIDATION CHECK
# ==========================================
print("🚀 Initializing LRU Cache Engine with Capacity = 2")
memory_cache = LRUCache(2)

print("-> Writing [Key 1: Model Weights Alpha]")
memory_cache.put(1, 100)

print("-> Writing [Key 2: Model Weights Beta]")
memory_cache.put(2, 200)

print(f"-> Fetching Key 1: Data value returned = {memory_cache.get(1)} (Promoted to Head!)")

print("-> Writing [Key 3: Model Weights Gamma] -> Triggering Capacity Eviction!")
memory_cache.put(3, 300)

print("\n--- INFRASTRUCTURE MEMORY STATE AUDIT ---")
print(f"Checking Key 2 (Oldest data): Value = {memory_cache.get(2)} (Successfully Evicted -> -1)")
print(f"Checking Key 1 (Active data): Value = {memory_cache.get(1)} (Still Safe in RAM -> 100)")
print(f"Checking Key 3 (Fresh data):  Value = {memory_cache.get(3)} (Still Safe in RAM -> 300)")
