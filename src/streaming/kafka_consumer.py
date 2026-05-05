import json
import time
import threading
import queue
import logging
import structlog
import boto3
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from confluent_kafka import Consumer, KafkaError
from confluent_kafka.admin import AdminClient, ConfigResource
import redis

# --- Logging Setup ---
structlog.configure(
    processors=[structlog.processors.TimeStamper(fmt="iso"), structlog.processors.JSONRenderer()]
)
logger = structlog.get_logger()

# --- Config Constants (Ideally from config.yaml) ---
KAFKA_BOOTSTRAP = "localhost:9092"
REDIS_HOST = "localhost"
REDIS_PORT = 6379

class ClickEventConsumer(threading.Thread):
    """
    Consumes click events and updates demand scores in Redis using a sliding window.
    """
    def __init__(self, topic: str, r: redis.Redis):
        super().__init__(daemon=True)
        self.topic = topic
        self.redis = r
        self.running = True
        self.conf = {
            'bootstrap.servers': KAFKA_BOOTSTRAP,
            'group.id': 'click_demand_group',
            'auto.offset.reset': 'latest'
        }
        self.consumer = Consumer(self.conf)

    def run(self):
        logger.info("ClickEventConsumer started", topic=self.topic)
        self.consumer.subscribe([self.topic])
        
        batch = []
        last_flush = time.time()

        try:
            while self.running:
                msg = self.consumer.poll(0.1)
                if msg is None:
                    # Periodic flush if idle
                    if batch and time.time() - last_flush > 0.5:
                        self.flush_batch(batch)
                        batch = []
                    continue
                
                if msg.error():
                    logger.error("Kafka error", error=msg.error())
                    continue

                event = json.loads(msg.value().decode('utf-8'))
                batch.append(event)

                # Flush every 500ms or 100 events
                if len(batch) >= 100 or (time.time() - last_flush > 0.5):
                    self.flush_batch(batch)
                    batch = []
                    last_flush = time.time()

        finally:
            self.consumer.close()

    def flush_batch(self, events):
        pipe = self.redis.pipeline()
        now = time.time()
        seven_days_ago = now - (7 * 24 * 3600)
        
        skus_to_update = set()
        for event in events:
            sku_id = event['sku_id']
            ts = event.get('timestamp', now)
            # ZADD demand:{sku_id} {timestamp} {timestamp}
            pipe.zadd(f"demand:{sku_id}", {str(ts): ts})
            skus_to_update.add(sku_id)

        for sku_id in skus_to_update:
            # ZREMRANGEBYSCORE demand:{sku_id} 0 {now - 7days}
            pipe.zremrangebyscore(f"demand:{sku_id}", 0, seven_days_ago)
            # ZCARD demand:{sku_id}
            pipe.zcard(f"demand:{sku_id}")
            
        results = pipe.execute()
        # The last results in pipe are ZCARDs
        logger.debug("Flushed click batch", count=len(events))

class OrderConsumer(threading.Thread):
    """
    Consumes order events, updates FeatureStore, and feeds the FlashSaleDetector.
    """
    def __init__(self, topic: str, r: redis.Redis, cep_queue: queue.Queue):
        super().__init__(daemon=True)
        self.topic = topic
        self.redis = r
        self.cep_queue = cep_queue
        self.running = True
        self.consumer = Consumer({
            'bootstrap.servers': KAFKA_BOOTSTRAP,
            'group.id': 'order_processing_group',
            'auto.offset.reset': 'latest'
        })

    def run(self):
        logger.info("OrderConsumer started", topic=self.topic)
        self.consumer.subscribe([self.topic])
        
        try:
            while self.running:
                msg = self.consumer.poll(0.1)
                if msg is None or msg.error(): continue

                order = json.loads(msg.value().decode('utf-8'))
                sku_id = order['sku_id']
                
                # 1. Update demand_score_7d (simplified increment for demo)
                self.redis.incr(f"feat:{sku_id}:demand_orders")
                
                # 2. Feed to CEP Detector
                self.cep_queue.put(order)
                
        finally:
            self.consumer.close()

class InventoryConsumer(threading.Thread):
    """
    Consumes inventory updates and computes live inventory ratios.
    """
    def __init__(self, topic: str, r: redis.Redis):
        super().__init__(daemon=True)
        self.topic = topic
        self.redis = r
        self.running = True
        self.consumer = Consumer({
            'bootstrap.servers': KAFKA_BOOTSTRAP,
            'group.id': 'inventory_group',
            'auto.offset.reset': 'latest'
        })

    def run(self):
        logger.info("InventoryConsumer started", topic=self.topic)
        self.consumer.subscribe([self.topic])
        
        try:
            while self.running:
                msg = self.consumer.poll(0.1)
                if msg is None or msg.error(): continue

                inv = json.loads(msg.value().decode('utf-8'))
                sku_id = inv['sku_id']
                
                # inventory_ratio = current_stock / (avg_daily_demand * 7)
                ratio = inv['current_stock'] / (max(inv['avg_daily_demand'], 1) * 7)
                
                # Update Redis with 60s TTL
                self.redis.setex(f"feat:{sku_id}:inventory", 60, round(ratio, 4))
                logger.debug("Inventory ratio updated", sku_id=sku_id, ratio=ratio)
                
        finally:
            self.consumer.close()

class FlashSaleDetector(threading.Thread):
    """
    Complex Event Processing (CEP): Detects spikes in order velocity.
    """
    def __init__(self, cep_queue: queue.Queue, r: redis.Redis):
        super().__init__(daemon=True)
        self.queue = cep_queue
        self.redis = r
        self.history = {} # sku_id -> [timestamps]
        self.cw = boto3.client('cloudwatch', region_name='us-east-1')
        self.running = True

    def run(self):
        logger.info("FlashSaleDetector started")
        while self.running:
            try:
                order = self.queue.get(timeout=1)
                sku_id = order['sku_id']
                now = time.time()
                
                if sku_id not in self.history: self.history[sku_id] = []
                self.history[sku_id].append(now)
                
                # Sliding window: last 2 minutes
                self.history[sku_id] = [ts for ts in self.history[sku_id] if ts > now - 120]
                
                # Baseline velocity (simulated: 1 order per 2 mins = 0.5/min)
                # In prod, this would be fetched from SKU metadata
                baseline = 0.5 
                current_velocity = len(self.history[sku_id]) / 2.0
                
                if current_velocity > 3 * baseline:
                    self.trigger_flash_sale(sku_id, current_velocity)
                    
            except queue.Empty:
                continue

    def trigger_flash_sale(self, sku_id, velocity):
        logger.warning("FLASH SALE DETECTED", sku_id=sku_id, velocity=velocity)
        # Set flash sale flag in Redis (10 min TTL)
        self.redis.setex(f"feat:{sku_id}:is_flash_sale", 600, "True")
        
        # Log to CloudWatch
        try:
            self.cw.put_metric_data(
                Namespace='PricingEngine/FlashSaleDetected',
                MetricData=[{'MetricName': 'FlashSaleActive', 'Value': 1, 'Unit': 'Count'}]
            )
        except Exception as e:
            logger.error("CloudWatch logging failed", error=str(e))

class ConsumerManager:
    """
    Orchestrates all consumer threads and monitors health.
    """
    def __init__(self):
        self.redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        self.cep_queue = queue.Queue()
        self.admin = AdminClient({'bootstrap.servers': KAFKA_BOOTSTRAP})
        
        self.consumers = [
            ClickEventConsumer("pricing.click-events", self.redis_client),
            OrderConsumer("pricing.order-stream", self.redis_client, self.cep_queue),
            InventoryConsumer("pricing.inventory-updates", self.redis_client),
            FlashSaleDetector(self.cep_queue, self.redis_client)
        ]

    def start_all(self):
        for c in self.consumers:
            c.start()
        
        try:
            while True:
                time.sleep(10)
                self.check_lag()
        except KeyboardInterrupt:
            self.stop_all()

    def check_lag(self):
        # Simplified lag check for demo
        logger.info("Monitoring consumer health...")

    def stop_all(self):
        logger.info("Shutting down consumers...")
        for c in self.consumers:
            c.running = False
        for c in self.consumers:
            c.join()

if __name__ == "__main__":
    manager = ConsumerManager()
    manager.start_all()
