"""
streaming/producer.py — Kafka event producer for UCI retail order events.

Publishes synthetic order events to the order_events topic.
Falls back to a simple print-based simulation when Kafka is unavailable.
"""
import sys
import json
import time
import random
from pathlib import Path
from datetime import datetime

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

from config import KAFKA_BROKER, KAFKA_ORDER_TOPIC

# Sample stock codes from the UCI dataset
SAMPLE_PRODUCTS = [
    ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 2.55),
    ("71053" , "WHITE METAL LANTERN"                , 3.39),
    ("84406B", "CREAM CUPID HEARTS COAT HANGER"     , 2.75),
    ("84029G", "KNITTED UNION FLAG HOT WATER BOTTLE" , 3.39),
    ("84029E", "RED WOOLLY HOTTIE WHITE HEART"        , 3.39),
    ("22752" , "SET 7 BABUSHKA NESTING BOXES"         , 7.65),
    ("21730" , "GLASS STAR FROSTED T-LIGHT HOLDER"    , 4.25),
    ("22633" , "HAND WARMER UNION JACK"               , 1.85),
]


def _make_order_event() -> dict:
    code, desc, base_price = random.choice(SAMPLE_PRODUCTS)
    qty = random.randint(1, 48)
    price_jitter = base_price * random.uniform(0.9, 1.1)
    return {
        "event_type"    : "order",
        "timestamp"     : datetime.utcnow().isoformat(),
        "invoice_no"    : f"INV{random.randint(500000, 599999)}",
        "stock_code"    : code,
        "description"   : desc,
        "quantity"      : qty,
        "unit_price"    : round(price_jitter, 4),
        "revenue"       : round(qty * price_jitter, 4),
        "customer_id"   : f"CUST_{random.randint(10000, 18999)}",
        "country"       : random.choice(["United Kingdom"] * 8 + ["Germany", "France", "Netherlands"]),
    }


def run_producer(n_events: int = 100, delay: float = 0.5):
    """Publish n_events order events to Kafka, sleeping delay seconds between each."""
    try:
        from kafka import KafkaProducer
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BROKER,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        kafka_ok = True
    except Exception as e:
        print(f"[Producer] Kafka unavailable ({e}) — printing events to stdout")
        kafka_ok = False

    for i in range(n_events):
        event = _make_order_event()
        if kafka_ok:
            producer.send(KAFKA_ORDER_TOPIC, value=event)
            if (i + 1) % 10 == 0:
                producer.flush()
        else:
            print(f"[Event {i+1:03d}] {json.dumps(event)}")
        time.sleep(delay)

    if kafka_ok:
        producer.flush()
        producer.close()
    print(f"[Producer] Sent {n_events} events")


if __name__ == "__main__":
    run_producer(n_events=20, delay=0.2)
