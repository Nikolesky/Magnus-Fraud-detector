import pandas as pd
import datetime as dt
from collections import defaultdict, deque, Counter
import heapq
import bisect
import matplotlib.pyplot as plt
import seaborn as sns

# ========================================
# DATA STRUCTURE-BASED ANALYSIS TOOLKIT
# ========================================

# -------------------------------
# 1. HASH MAP: User Statistics Analyzer
# -------------------------------
class UserStatsAnalyzer:
    """Hash map for O(1) user statistics lookup and analysis"""
    def __init__(self, df):
        self.user_stats = defaultdict(lambda: {
            'total_sent': 0, 'total_received': 0,
            'fraud_sent': 0, 'fraud_received': 0,
            'num_sent': 0, 'num_received': 0,
            'partners': set(), 'fraud_partners': set()
        })
        
        print("🔨 Building hash map index...")
        self._build_index(df)
    
    def _build_index(self, df):
        """O(n) build, then O(1) lookups"""
        for _, row in df.iterrows():
            sender, receiver = row['sender'], row['receiver']
            amount, is_fraud = row['amount'], row['is_fraud']
            
            # Sender stats
            self.user_stats[sender]['total_sent'] += amount
            self.user_stats[sender]['num_sent'] += 1
            self.user_stats[sender]['partners'].add(receiver)
            if is_fraud:
                self.user_stats[sender]['fraud_sent'] += amount
                self.user_stats[sender]['fraud_partners'].add(receiver)
            
            # Receiver stats
            self.user_stats[receiver]['total_received'] += amount
            self.user_stats[receiver]['num_received'] += 1
            if is_fraud:
                self.user_stats[receiver]['fraud_received'] += amount
    
    def get_user_stats(self, user_id):
        """O(1) lookup"""
        return self.user_stats[user_id]
    
    def find_top_fraudsters(self, n=10):
        """Find users with most fraud transactions - O(n log k) using heap"""
        fraud_scores = []
        for user, stats in self.user_stats.items():
            fraud_ratio = 0
            if stats['num_sent'] > 0:
                fraud_ratio = stats['fraud_sent'] / stats['total_sent']
            if fraud_ratio > 0:
                heapq.heappush(fraud_scores, (-fraud_ratio, user, stats['fraud_sent']))
        
        top = heapq.nsmallest(n, fraud_scores)
        print(f"\n📊 Top {n} Fraudsters (Hash Map Analysis):")
        for i, (neg_ratio, user, fraud_amt) in enumerate(top, 1):
            print(f"   {i}. {user}: {-neg_ratio:.2%} fraud ratio, ${fraud_amt:,.2f} fraudulent")
        return top
    
    def find_high_velocity_users(self, threshold=100):
        """Users with unusually high transaction counts"""
        high_velocity = [(u, s['num_sent']) for u, s in self.user_stats.items() 
                        if s['num_sent'] > threshold]
        high_velocity.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n⚡ High Velocity Users (>{threshold} transactions):")
        for user, count in high_velocity[:10]:
            print(f"   {user}: {count} transactions")
        return high_velocity


# -------------------------------
# 2. GRAPH: Network Analysis with Adjacency List
# -------------------------------
class TransactionGraphAnalyzer:
    """Directed graph using adjacency list for network analysis"""
    def __init__(self, df):
        self.adj_list = defaultdict(list)  # sender -> [(receiver, amount, is_fraud, timestamp)]
        self.reverse_adj = defaultdict(list)  # receiver -> [sender, ...]
        self.in_degree = Counter()
        self.out_degree = Counter()
        
        print("🔨 Building graph (adjacency list)...")
        self._build_graph(df)
    
    def _build_graph(self, df):
        """Build directed graph from transactions"""
        for _, row in df.iterrows():
            sender, receiver = row['sender'], row['receiver']
            amount, is_fraud = row['amount'], row['is_fraud']
            timestamp = row['timestamp']
            
            self.adj_list[sender].append((receiver, amount, is_fraud, timestamp))
            self.reverse_adj[receiver].append((sender, amount, is_fraud, timestamp))
            self.out_degree[sender] += 1
            self.in_degree[receiver] += 1
    
    def detect_cycles(self, start_user, max_length=5):
        """BFS to find circular money flows (potential money laundering)"""
        cycles = []
        queue = deque([(start_user, [start_user], 0)])
        visited_paths = set()
        
        while queue:
            current, path, length = queue.popleft()
            
            if length >= max_length:
                continue
            
            for neighbor, amount, is_fraud, _ in self.adj_list[current]:
                if neighbor == start_user and length > 0:
                    cycle = tuple(path + [neighbor])
                    if cycle not in visited_paths:
                        cycles.append((cycle, length + 1))
                        visited_paths.add(cycle)
                elif neighbor not in path:
                    queue.append((neighbor, path + [neighbor], length + 1))
        
        return cycles
    
    def find_money_laundering_patterns(self, sample_size=50):
        """Detect circular transaction patterns"""
        print(f"\n🔍 Detecting Money Laundering Cycles (BFS on Graph)...")
        all_cycles = []
        
        # Sample users to check
        users_to_check = list(self.adj_list.keys())[:sample_size]
        
        for user in users_to_check:
            cycles = self.detect_cycles(user, max_length=4)
            all_cycles.extend(cycles)
        
        print(f"   Found {len(all_cycles)} potential circular flows")
        if all_cycles:
            print(f"   Sample cycles:")
            for cycle, length in all_cycles[:5]:
                print(f"      {' -> '.join(cycle)} (length: {length})")
        
        return all_cycles
    
    def find_hubs(self, n=10):
        """Find most connected nodes in the graph"""
        print(f"\n🌐 Network Hubs (Graph Degree Analysis):")
        
        # Out-degree (senders)
        top_senders = self.out_degree.most_common(n)
        print(f"   Top Senders (out-degree):")
        for user, degree in top_senders:
            print(f"      {user}: {degree} outgoing transactions")
        
        # In-degree (receivers)
        top_receivers = self.in_degree.most_common(n)
        print(f"\n   Top Receivers (in-degree):")
        for user, degree in top_receivers:
            print(f"      {user}: {degree} incoming transactions")
        
        return top_senders, top_receivers
    
    def connected_components_dfs(self):
        """Find connected components using DFS (undirected view)"""
        visited = set()
        components = []
        
        all_users = set(self.adj_list.keys()) | set(self.reverse_adj.keys())
        
        def dfs(node, component):
            visited.add(node)
            component.add(node)
            
            # Check outgoing edges
            for neighbor, _, _, _ in self.adj_list[node]:
                if neighbor not in visited:
                    dfs(neighbor, component)
            
            # Check incoming edges
            for neighbor, _, _, _ in self.reverse_adj[node]:
                if neighbor not in visited:
                    dfs(neighbor, component)
        
        for user in all_users:
            if user not in visited:
                component = set()
                dfs(user, component)
                components.append(component)
        
        print(f"\n🔗 Graph Connectivity (DFS):")
        print(f"   Found {len(components)} connected components")
        print(f"   Largest component: {len(max(components, key=len))} users")
        
        return components


# -------------------------------
# 3. SORTED LIST: Time-Series Analysis with Binary Search
# -------------------------------
class TimeSeriesAnalyzer:
    """Binary search tree (sorted list) for efficient time-range queries"""
    def __init__(self, df):
        self.sorted_transactions = []
        print("🔨 Building sorted transaction index (for binary search)...")
        self._build_sorted_index(df)
    
    def _build_sorted_index(self, df):
        """Build sorted list by timestamp - O(n log n)"""
        for _, row in df.iterrows():
            self.sorted_transactions.append((
                row['timestamp'],
                row['sender'],
                row['receiver'],
                row['amount'],
                row['is_fraud']
            ))
        self.sorted_transactions.sort(key=lambda x: x[0])
    
    def get_transactions_in_window(self, start_time, end_time):
        """Binary search for time range - O(log n + k) where k is results"""
        # Find start index
        start_idx = bisect.bisect_left(
            self.sorted_transactions, 
            (start_time, '', '', 0, 0)
        )
        
        # Find end index
        end_idx = bisect.bisect_right(
            self.sorted_transactions,
            (end_time, 'ZZZZ', 'ZZZZ', float('inf'), 1)
        )
        
        return self.sorted_transactions[start_idx:end_idx]
    
    def analyze_hourly_patterns(self):
        """Analyze fraud patterns by hour of day"""
        print(f"\n🕐 Hourly Fraud Pattern Analysis (Binary Search):")
        
        hourly_fraud = defaultdict(lambda: {'total': 0, 'fraud': 0})
        
        for timestamp, _, _, amount, is_fraud in self.sorted_transactions:
            hour = timestamp.hour
            hourly_fraud[hour]['total'] += 1
            if is_fraud:
                hourly_fraud[hour]['fraud'] += 1
        
        print("   Hour | Total Txns | Frauds | Fraud %")
        print("   " + "-" * 45)
        for hour in sorted(hourly_fraud.keys()):
            stats = hourly_fraud[hour]
            fraud_pct = (stats['fraud'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"   {hour:02d}:00 | {stats['total']:10d} | {stats['fraud']:6d} | {fraud_pct:6.2f}%")
        
        return hourly_fraud
    
    def find_burst_periods(self, window_minutes=60, threshold=100):
        """Find time windows with unusually high transaction volume"""
        print(f"\n💥 Burst Detection (Sliding Window with Binary Search):")
        
        bursts = []
        
        # Sample time points to check
        for i in range(0, len(self.sorted_transactions), 1000):
            start_time = self.sorted_transactions[i][0]
            end_time = start_time + dt.timedelta(minutes=window_minutes)
            
            window_txns = self.get_transactions_in_window(start_time, end_time)
            
            if len(window_txns) > threshold:
                fraud_count = sum(1 for _, _, _, _, is_fraud in window_txns if is_fraud)
                bursts.append((start_time, len(window_txns), fraud_count))
        
        bursts.sort(key=lambda x: x[1], reverse=True)
        
        print(f"   Found {len(bursts)} burst periods (>{threshold} txns in {window_minutes}min)")
        for i, (time, count, frauds) in enumerate(bursts[:5], 1):
            print(f"   {i}. {time}: {count} transactions ({frauds} frauds)")
        
        return bursts


# -------------------------------
# 4. PRIORITY QUEUE: Top-K Analysis
# -------------------------------
class TopKAnalyzer:
    """Min-heap for efficient top-K queries"""
    def __init__(self, df):
        self.df = df
    
    def find_largest_transactions(self, k=20):
        """Find K largest transactions using max-heap"""
        print(f"\n💰 Top {k} Largest Transactions (Max-Heap):")
        
        heap = []
        for _, row in self.df.iterrows():
            heapq.heappush(heap, (-row['amount'], row['sender'], row['receiver'], row['is_fraud']))
        
        top_k = heapq.nsmallest(k, heap)
        
        for i, (neg_amount, sender, receiver, is_fraud) in enumerate(top_k, 1):
            fraud_label = "🚨 FRAUD" if is_fraud else "✓ Normal"
            print(f"   {i}. ${-neg_amount:,.2f} | {sender} → {receiver} | {fraud_label}")
        
        return top_k
    
    def find_suspicious_amount_patterns(self):
        """Find suspiciously round amounts (e.g., exactly 10000.00)"""
        print(f"\n🎯 Suspicious Round Amounts (Hash Set):")
        
        round_amounts = defaultdict(int)
        for _, row in self.df.iterrows():
            amount = row['amount']
            if amount % 1000 == 0 or amount % 5000 == 0:  # Round thousands
                round_amounts[amount] += 1
        
        # Use heap to get top repeated round amounts
        top_rounds = heapq.nlargest(10, 
            [(count, amount) for amount, count in round_amounts.items()])
        
        for count, amount in top_rounds:
            print(f"   ${amount:,.0f}: appears {count} times")
        
        return top_rounds


# -------------------------------
# 5. TRIE: Pattern Matching
# -------------------------------
class PatternMatcher:
    """Trie for detecting repeated transaction sequences"""
    def __init__(self):
        self.root = {}
        self.pattern_counts = Counter()
    
    def insert_sequence(self, sequence):
        """Insert a transaction sequence into trie"""
        node = self.root
        for item in sequence:
            if item not in node:
                node[item] = {}
            node = node[item]
        node['*'] = node.get('*', 0) + 1
        self.pattern_counts[sequence] += 1
    
    def analyze_patterns(self, df, sequence_length=3):
        """Find repeated transaction patterns"""
        print(f"\n🔍 Repeated Transaction Patterns (Trie):")
        
        # Group by user and build sequences
        user_sequences = defaultdict(list)
        for _, row in df.iterrows():
            user_sequences[row['sender']].append((row['receiver'], row['amount'], row['is_fraud']))
        
        # Extract patterns
        for user, txns in user_sequences.items():
            for i in range(len(txns) - sequence_length + 1):
                sequence = tuple(txns[i:i+sequence_length])
                self.insert_sequence(sequence)
        
        # Find most common patterns
        top_patterns = self.pattern_counts.most_common(10)
        
        print(f"   Found {len(self.pattern_counts)} unique patterns")
        print(f"   Most repeated patterns:")
        for i, (pattern, count) in enumerate(top_patterns, 1):
            if count > 1:
                print(f"   {i}. Pattern repeated {count} times: {pattern[0][:2]}...")
        
        return top_patterns


# -------------------------------
# MAIN ANALYSIS RUNNER
# -------------------------------
def run_comprehensive_analysis(csv_file="synthetic_transactions50.csv"):
    """Run all analyses on the dataset"""
    print("=" * 60)
    print("FRAUD DETECTION DATA ANALYSIS")
    print("Using Advanced Data Structures")
    print("=" * 60)
    
    # Load data
    print(f"\n📂 Loading {csv_file}...")
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"   Loaded {len(df):,} transactions")
    print(f"   Fraud rate: {df['is_fraud'].mean()*100:.2f}%")
    
    # 1. Hash Map Analysis
    print("\n" + "="*60)
    print("1️⃣  HASH MAP ANALYSIS")
    print("="*60)
    user_analyzer = UserStatsAnalyzer(df)
    user_analyzer.find_top_fraudsters(n=10)
    user_analyzer.find_high_velocity_users(threshold=100)
    
    # 2. Graph Analysis
    print("\n" + "="*60)
    print("2️⃣  GRAPH ANALYSIS (Adjacency List)")
    print("="*60)
    graph_analyzer = TransactionGraphAnalyzer(df)
    graph_analyzer.find_hubs(n=10)
    graph_analyzer.find_money_laundering_patterns(sample_size=30)
    graph_analyzer.connected_components_dfs()
    
    # 3. Time Series Analysis
    print("\n" + "="*60)
    print("3️⃣  TIME-SERIES ANALYSIS (Binary Search)")
    print("="*60)
    time_analyzer = TimeSeriesAnalyzer(df)
    time_analyzer.analyze_hourly_patterns()
    time_analyzer.find_burst_periods(window_minutes=60, threshold=50)
    
    # 4. Priority Queue Analysis
    print("\n" + "="*60)
    print("4️⃣  TOP-K ANALYSIS (Heap)")
    print("="*60)
    topk_analyzer = TopKAnalyzer(df)
    topk_analyzer.find_largest_transactions(k=15)
    topk_analyzer.find_suspicious_amount_patterns()
    
    # 5. Pattern Matching
    print("\n" + "="*60)
    print("5️⃣  PATTERN MATCHING (Trie)")
    print("="*60)
    pattern_matcher = PatternMatcher()
    pattern_matcher.analyze_patterns(df, sequence_length=3)
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE")
    print("="*60)


def export_results_to_files(user_analyzer, graph_analyzer, time_analyzer, topk_analyzer, pattern_matcher, df):
    """Export all analysis results to various file formats"""
    import json
    from datetime import datetime
    
    print("\n📁 Exporting results to files...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. JSON - Structured results
    results = {
        "analysis_timestamp": timestamp,
        "dataset_stats": {
            "total_transactions": len(df),
            "fraud_count": int(df['is_fraud'].sum()),
            "fraud_percentage": float(df['is_fraud'].mean() * 100),
            "unique_users": len(set(df['sender'].unique()) | set(df['receiver'].unique()))
        },
        "top_fraudsters": [],
        "network_hubs": {},
        "hourly_patterns": {},
        "top_transactions": []
    }
    
    # Add top fraudsters
    for neg_ratio, user, fraud_amt in user_analyzer.find_top_fraudsters(10):
        results["top_fraudsters"].append({
            "user_id": user,
            "fraud_ratio": float(-neg_ratio),
            "fraud_amount": float(fraud_amt)
        })
    
    with open(f"analysis_results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"   ✓ JSON: analysis_results_{timestamp}.json")
    
    # 2. CSV - User statistics
    user_stats_data = []
    for user, stats in user_analyzer.user_stats.items():
        user_stats_data.append({
            'user_id': user,
            'total_sent': stats['total_sent'],
            'total_received': stats['total_received'],
            'num_sent': stats['num_sent'],
            'num_received': stats['num_received'],
            'fraud_sent': stats['fraud_sent'],
            'fraud_received': stats['fraud_received'],
            'num_partners': len(stats['partners']),
            'num_fraud_partners': len(stats['fraud_partners'])
        })
    
    user_stats_df = pd.DataFrame(user_stats_data)
    user_stats_df.to_csv(f"user_statistics_{timestamp}.csv", index=False)
    print(f"   ✓ CSV: user_statistics_{timestamp}.csv")
    
    # 3. CSV - Network metrics
    network_data = []
    for user in graph_analyzer.adj_list.keys():
        network_data.append({
            'user_id': user,
            'out_degree': graph_analyzer.out_degree[user],
            'in_degree': graph_analyzer.in_degree[user],
            'total_degree': graph_analyzer.out_degree[user] + graph_analyzer.in_degree[user]
        })
    
    network_df = pd.DataFrame(network_data)
    network_df = network_df.sort_values('total_degree', ascending=False)
    network_df.to_csv(f"network_metrics_{timestamp}.csv", index=False)
    print(f"   ✓ CSV: network_metrics_{timestamp}.csv")
    
    # 4. TXT - Detailed report
    with open(f"analysis_report_{timestamp}.txt", "w") as f:
        f.write("="*70 + "\n")
        f.write("FRAUD DETECTION ANALYSIS REPORT\n")
        f.write(f"Generated: {timestamp}\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Dataset Statistics:\n")
        f.write(f"  Total Transactions: {len(df):,}\n")
        f.write(f"  Fraudulent: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)\n")
        f.write(f"  Normal: {(~df['is_fraud'].astype(bool)).sum():,}\n\n")
        
        f.write("Top 10 Fraudsters:\n")
        for i, (neg_ratio, user, fraud_amt) in enumerate(user_analyzer.find_top_fraudsters(10), 1):
            f.write(f"  {i}. {user}: {-neg_ratio:.2%} fraud ratio, ${fraud_amt:,.2f}\n")
        
        f.write("\nNetwork Hubs (Top 5 by degree):\n")
        top_senders, top_receivers = graph_analyzer.find_hubs(5)
        f.write("  Senders:\n")
        for user, degree in top_senders:
            f.write(f"    {user}: {degree} transactions\n")
    
    print(f"   ✓ TXT: analysis_report_{timestamp}.txt")
    
    # 5. CSV - Suspicious transactions (large amounts)
    suspicious = df.nlargest(100, 'amount')[['sender', 'receiver', 'amount', 'timestamp', 'is_fraud']]
    suspicious.to_csv(f"suspicious_transactions_{timestamp}.csv", index=False)
    print(f"   ✓ CSV: suspicious_transactions_{timestamp}.csv")
    
    print(f"\n✅ All results exported with timestamp: {timestamp}")


if __name__ == "__main__":
    # Run analysis on your generated dataset
    print("=" * 60)
    print("FRAUD DETECTION DATA ANALYSIS")
    print("Using Advanced Data Structures")
    print("=" * 60)
    
    # Load data
    csv_file = "synthetic_transactions50.csv"
    print(f"\n📂 Loading {csv_file}...")
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"   Loaded {len(df):,} transactions")
    print(f"   Fraud rate: {df['is_fraud'].mean()*100:.2f}%")
    
    # 1. Hash Map Analysis
    print("\n" + "="*60)
    print("1️⃣  HASH MAP ANALYSIS")
    print("="*60)
    user_analyzer = UserStatsAnalyzer(df)
    user_analyzer.find_top_fraudsters(n=10)
    user_analyzer.find_high_velocity_users(threshold=100)
    
    # 2. Graph Analysis
    print("\n" + "="*60)
    print("2️⃣  GRAPH ANALYSIS (Adjacency List)")
    print("="*60)
    graph_analyzer = TransactionGraphAnalyzer(df)
    graph_analyzer.find_hubs(n=10)
    graph_analyzer.find_money_laundering_patterns(sample_size=30)
    graph_analyzer.connected_components_dfs()
    
    # 3. Time Series Analysis
    print("\n" + "="*60)
    print("3️⃣  TIME-SERIES ANALYSIS (Binary Search)")
    print("="*60)
    time_analyzer = TimeSeriesAnalyzer(df)
    time_analyzer.analyze_hourly_patterns()
    time_analyzer.find_burst_periods(window_minutes=60, threshold=50)
    
    # 4. Priority Queue Analysis
    print("\n" + "="*60)
    print("4️⃣  TOP-K ANALYSIS (Heap)")
    print("="*60)
    topk_analyzer = TopKAnalyzer(df)
    topk_analyzer.find_largest_transactions(k=15)
    topk_analyzer.find_suspicious_amount_patterns()
    
    # 5. Pattern Matching
    print("\n" + "="*60)
    print("5️⃣  PATTERN MATCHING (Trie)")
    print("="*60)
    pattern_matcher = PatternMatcher()
    pattern_matcher.analyze_patterns(df, sequence_length=3)
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE")
    print("="*60)
    
    # Export results
    export_results_to_files(user_analyzer, graph_analyzer, time_analyzer, topk_analyzer, pattern_matcher, df)