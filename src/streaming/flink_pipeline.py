import json
import argparse
import time
import requests
from typing import Iterable, List
from pyflink.common import WatermarkStrategy, Time, Configuration
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types
from pyflink.datastream import StreamExecutionEnvironment, RuntimeExecutionMode
from pyflink.datastream.connectors.kafka import FlinkKafkaConsumer
from pyflink.datastream.window import SlidingProcessingTimeWindows
from pyflink.datastream.functions import ProcessWindowFunction, RuntimeContext
from pyflink.datastream.state import ValueStateDescriptor

# --- Configuration & Models ---
API_BATCH_URL = "http://localhost:8000/api/v1/price/batch"
API_CACHE_URL = "http://localhost:8000/api/v1/price/{}/cache"

class DemandAggregate:
    def __init__(self, sku_id, window_end, order_count, revenue, avg_price):
        self.sku_id = sku_id
        self.window_end = window_end
        self.order_count = order_count
        self.revenue = revenue
        self.avg_price = avg_price

    def to_json(self):
        return json.dumps(self.__dict__)

# --- Processing Functions ---

class AggregateWindowFunction(ProcessWindowFunction[Iterable[str], str, str, SlidingProcessingTimeWindows]):
    """
    Computes count, revenue and avg price for a 5-min sliding window.
    """
    def process(self, key: str, context: 'ProcessWindowFunction.Context', elements: Iterable[str]) -> Iterable[str]:
        count = 0
        revenue = 0.0
        for msg in elements:
            try:
                data = json.loads(msg)
                count += 1
                revenue += data.get('price', 0.0)
            except:
                continue
        
        avg_price = revenue / count if count > 0 else 0.0
        agg = DemandAggregate(key, context.window().end, count, revenue, avg_price)
        
        # Trigger Price Sync in Backend (Batch API)
        try:
            # Simplified: In prod this would be an async HTTP call or a dedicated Sink
            requests.post(API_BATCH_URL, json={"items": [{"sku_id": key, "demand_score": count}]})
        except:
            pass
            
        yield agg.to_json()

class FlashSaleDetector(ProcessWindowFunction):
    """
    Stateful detector for order spikes.
    Logic: If current window count > 3 * baseline (previous windows), trigger Flash Sale.
    """
    def open(self, runtime_context: RuntimeContext):
        # State to track baseline order counts
        descriptor = ValueStateDescriptor("baseline_velocity", Types.DOUBLE())
        self.baseline_state = runtime_context.get_state(descriptor)

    def process(self, key: str, context: 'ProcessWindowFunction.Context', elements: Iterable[str]) -> Iterable[str]:
        # Count elements in current window
        current_count = sum(1 for _ in elements)
        
        # Get baseline from state
        baseline = self.baseline_state.value()
        if baseline is None:
            # First time seeing this SKU, initialize baseline
            self.baseline_state.update(float(current_count))
            yield json.dumps({"sku_id": key, "spike_detected": False, "count": current_count})
            return

        # Spike Detection: If current > 3x baseline
        is_spike = current_count > 3 * baseline
        if is_spike:
            # In production, this could trigger a webhook or update a Redis flag
            print(f"!!! FLASH SALE SPIKE for {key}: {current_count} (Baseline: {baseline:.2f})")
        
        # Update baseline (Exponential Moving Average)
        new_baseline = 0.8 * baseline + 0.2 * current_count
        self.baseline_state.update(new_baseline)
        
        yield json.dumps({
            "sku_id": key, 
            "spike_detected": is_spike, 
            "count": current_count,
            "baseline": round(new_baseline, 2),
            "timestamp": context.window().end
        })

# --- Main Pipeline ---

def run_flink_pipeline(is_local: bool):
    # 1. Setup Environment
    config = Configuration()
    env = StreamExecutionEnvironment.get_execution_environment(config)
    env.set_runtime_mode(RuntimeExecutionMode.STREAMING)
    
    # Checkpointing Configuration
    env.enable_checkpointing(30000) # 30 seconds
    checkpoint_config = env.get_checkpoint_config()
    checkpoint_config.set_checkpoint_storage("s3://olist-pricing-engine/checkpoints")
    
    # 2. Source Selection
    if is_local:
        logger_info = "Running Flink in LOCAL mode with synthetic source."
        # Simulated source for testing
        ds = env.from_collection(
            collection=[
                json.dumps({"sku_id": "SKU_A", "price": 100.0, "timestamp": time.time()}) for _ in range(100)
            ],
            type_info=Types.STRING()
        )
    else:
        # Kafka Source
        kafka_props = {'bootstrap.servers': 'localhost:9092', 'group.id': 'flink_pricing_group'}
        consumer = FlinkKafkaConsumer(
            topics=['pricing.order-stream', 'pricing.click-events'],
            deserialization_schema=SimpleStringSchema(),
            properties=kafka_props
        )
        ds = env.add_source(consumer)

    # 3. Aggregation Pipeline
    parsed_stream = ds \
        .map(lambda x: (json.loads(x)['sku_id'], x), output_type=Types.TUPLE([Types.STRING(), Types.STRING()])) \
        .key_by(lambda x: x[0])
        
    agg_stream = parsed_stream \
        .window(SlidingProcessingTimeWindows.of(Time.minutes(5), Time.minutes(1))) \
        .process(AggregateWindowFunction(), output_type=Types.STRING())

    # 3b. Flash Sale Detection Pipeline (CEP logic)
    flash_sale_stream = parsed_stream \
        .window(SlidingProcessingTimeWindows.of(Time.minutes(5), Time.minutes(1))) \
        .process(FlashSaleDetector(), output_type=Types.STRING())

    # 4. Sink: Log results
    agg_stream.print()
    flash_sale_stream.print()

    # 5. Execute
    print("Starting Flink Pricing Pipeline...")
    env.execute("Olist-Dynamic-Pricing-Engine")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Run with synthetic data source")
    args = parser.parse_args()
    
    run_flink_pipeline(args.local)
