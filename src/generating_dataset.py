import random
import pandas as pd
import datetime as dt

# -------------------------------
# CONFIGURATION
# -------------------------------
NUM_USERS = 1000
NUM_TRANSACTIONS = 50000
FRAUD_PERCENTAGE = 50   # 50% fraud
SEED = 42

random.seed(SEED)


# -------------------------------
# HELPERS
# -------------------------------
def random_timestamp(start_date, end_date):
    delta = end_date - start_date
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return start_date + dt.timedelta(seconds=random_seconds)


def generate_users(n):
    return [f"U{i}" for i in range(1, n + 1)]


def generate_transaction(users, start_date, end_date, is_fraud=0):
    sender = random.choice(users)
    receiver = random.choice([u for u in users if u != sender])
    amount = round(random.uniform(10, 20000), 2)
    timestamp = random_timestamp(start_date, end_date)
    txn_type = random.choice(["transfer", "cash_in", "payment"])
    return [sender, receiver, amount, timestamp, txn_type, is_fraud]


def inject_pattern_frauds(users, start, end, count=1000):
    """Optional realistic pattern frauds"""
    frauds = []
    for _ in range(count):
        sender = random.choice(users)
        receiver = random.choice([u for u in users if u != sender])
        amount = round(random.uniform(10000, 20000), 2)
        timestamp = random_timestamp(start, end)
        frauds.append([sender, receiver, amount, timestamp, "transfer", 1])
    return frauds


# -------------------------------
# MAIN GENERATOR
# -------------------------------
def generate_dataset(num_users=NUM_USERS, num_transactions=NUM_TRANSACTIONS, fraud_ratio=FRAUD_PERCENTAGE):
    users = generate_users(num_users)
    start, end = dt.datetime(2025, 10, 1), dt.datetime(2025, 10, 30)

    total_frauds = int(num_transactions * fraud_ratio / 100)
    total_normals = num_transactions - total_frauds

    print(f"🎯 Generating {total_frauds:,} fraudulent and {total_normals:,} normal transactions...")

    normal_data = [generate_transaction(users, start, end, 0) for _ in range(total_normals)]
    fraud_data = [generate_transaction(users, start, end, 1) for _ in range(total_frauds)]

    # Add some structured fraud patterns for realism
    fraud_data += inject_pattern_frauds(users, start, end, count=200)

    df = pd.DataFrame(normal_data + fraud_data,
                      columns=["sender", "receiver", "amount", "timestamp", "type", "is_fraud"])

    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    df.to_csv("synthetic_transactions.csv", index=False)
    print(f"✅ Generated dataset with {len(df)} rows ({df['is_fraud'].sum()} frauds ≈ {df['is_fraud'].mean()*100:.2f}%)")
    return df


# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    df = generate_dataset()
    print(df.head(10))
