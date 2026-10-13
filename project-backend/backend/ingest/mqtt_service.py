"""MQTT Ingestion Service for project-backend.
Subscribes via paho-mqtt v2 callback API with QoS 1.
Subscribes exclusively to: solar/+/readings, solar/+/health, solar/+/prediction, solar/+/twin.
Buffers incoming messages and flushes every 1 second or at 500 records.
"""
from datetime import datetime, timezone
import json
import os
import queue
import signal
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
import paho.mqtt.client as mqtt

from backend.config import verify_startup_safety
from backend.ingest.pipeline import get_counters, process_records
from backend.ingest.rejects import log_reject

load_dotenv()

# Limits
MAX_PAYLOAD_BYTES = 1024 * 1024  # 1 MB
MAX_BATCH_RECORDS = 5000

# Buffer settings
FLUSH_INTERVAL_S = 1.0
BUFFER_SIZE_FLUSH = 500

ALLOWED_STREAMS = {"readings", "health", "prediction", "twin"}


def handle_message(
    topic: str,
    payload_bytes: bytes,
    writer: Optional[Any] = None,
) -> Dict[str, Any]:
    """Plain function to process an incoming MQTT message payload.
    Testable without an active MQTT broker.

    Enforces:
    - Payload <= 1 MB (else TOO_LARGE)
    - Valid JSON (else BAD_JSON)
    - Array <= 5000 records (else TOO_LARGE)
    - Topic matching solar/{site_id}/{stream}
    """
    if len(payload_bytes) > MAX_PAYLOAD_BYTES:
        parts = topic.split("/")
        stream_cand = parts[2] if len(parts) == 3 and parts[0] == "solar" and parts[2] in ALLOWED_STREAMS else None
        log_reject(
            transport="mqtt",
            stream=stream_cand,
            code="TOO_LARGE",
            detail=f"Payload size {len(payload_bytes)} bytes exceeds 1 MB limit",
            record="<large payload>",
            topic=topic,
        )
        return {
            "accepted": 0,
            "rejected": 1,
            "written": 0,
            "write_failed": 0,
            "reasons": {"TOO_LARGE": 1},
        }

    parts = topic.split("/")
    if len(parts) != 3 or parts[0] != "solar":
        log_reject(
            transport="mqtt",
            stream=None,
            code="BAD_TOPIC",
            detail=f"Invalid topic structure: '{topic}'",
            record="",
            topic=topic,
        )
        return {
            "accepted": 0,
            "rejected": 1,
            "written": 0,
            "write_failed": 0,
            "reasons": {"BAD_TOPIC": 1},
        }

    site_id = parts[1]
    stream = parts[2]

    if stream not in ALLOWED_STREAMS:
        log_reject(
            transport="mqtt",
            stream=None,
            code="UNKNOWN_STREAM",
            detail=f"Stream '{stream}' is not supported",
            record="",
            topic=topic,
        )
        return {
            "accepted": 0,
            "rejected": 1,
            "written": 0,
            "write_failed": 0,
            "reasons": {"UNKNOWN_STREAM": 1},
        }

    try:
        data = json.loads(payload_bytes.decode("utf-8"))
    except Exception as e:
        log_reject(
            transport="mqtt",
            stream=stream,
            code="BAD_JSON",
            detail=f"Malformed JSON payload: {e}",
            record=payload_bytes[:200].decode("utf-8", errors="replace"),
            topic=topic,
        )
        return {
            "accepted": 0,
            "rejected": 1,
            "written": 0,
            "write_failed": 0,
            "reasons": {"BAD_JSON": 1},
        }

    if isinstance(data, list):
        if len(data) > MAX_BATCH_RECORDS:
            log_reject(
                transport="mqtt",
                stream=stream,
                code="TOO_LARGE",
                detail=f"Batch of {len(data)} exceeds {MAX_BATCH_RECORDS} records limit",
                record="",
                topic=topic,
            )
            return {
                "accepted": 0,
                "rejected": len(data),
                "written": 0,
                "write_failed": 0,
                "reasons": {"TOO_LARGE": len(data)},
            }
        records = data
    elif isinstance(data, dict):
        records = [data]
    else:
        log_reject(
            transport="mqtt",
            stream=stream,
            code="BAD_TYPE",
            detail=f"Payload must be JSON object or array, got {type(data).__name__}",
            record=str(data)[:200],
            topic=topic,
        )
        return {
            "accepted": 0,
            "rejected": 1,
            "written": 0,
            "write_failed": 0,
            "reasons": {"BAD_TYPE": 1},
        }

    return process_records(
        stream=stream,
        records=records,
        transport="mqtt",
        topic_site=site_id,
        writer=writer,
        topic=topic,
    )


class MqttIngestService:
    """Live MQTT ingestion daemon with background queue buffering."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        client_id: Optional[str] = None,
    ) -> None:
        verify_startup_safety()

        self.host = host or os.environ.get("MQTT_HOST", "localhost")
        self.port = int(port or os.environ.get("MQTT_PORT", 1883))
        self.client_id = client_id or os.environ.get("MQTT_CLIENT_ID", "solar-ingest")

        self.msg_queue: queue.Queue[Tuple[str, bytes]] = queue.Queue()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._stats_thread: Optional[threading.Thread] = None

        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=self.client_id,
        )
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            print(f"[MQTT] Failed to connect: reason={reason_code}", file=sys.stderr)
            return
        print(f"[MQTT] Connected to {self.host}:{self.port} as '{self.client_id}'")
        # Subscribe ONLY to valid project streams (never solar/+/fl-*)
        for s in ALLOWED_STREAMS:
            topic = f"solar/+/{s}"
            client.subscribe(topic, qos=1)
            print(f"[MQTT] Subscribed to {topic} (QoS 1)")

    def _on_message(self, client, userdata, message):
        self.msg_queue.put((message.topic, message.payload))

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties):
        print(f"[MQTT] Disconnected from broker: reason={reason_code}")

    def _worker_loop(self):
        while self._running or not self.msg_queue.empty():
            try:
                topic, payload = self.msg_queue.get(timeout=0.2)
                handle_message(topic, payload)
                self.msg_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[MQTT Worker Error] {e}", file=sys.stderr)

    def _stats_loop(self):
        while self._running:
            time.sleep(30)
            c = get_counters()
            print(
                f"[MQTT Ingest Counters] Received: {c['received']} | Accepted: {c['accepted']} | "
                f"Rejected: {c['rejected']} | Written: {c['written']} | Write Failed: {c['write_failed']}",
                flush=True,
            )

    def start(self):
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

        self._stats_thread = threading.Thread(target=self._stats_loop, daemon=True)
        self._stats_thread.start()

        print(f"[MQTT] Connecting to {self.host}:{self.port}...")
        self.client.connect(self.host, self.port, keepalive=60)
        self.client.loop_start()

    def stop(self):
        print("\n[MQTT] Stopping service and flushing queue...")
        self.client.loop_stop()
        self.client.disconnect()
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=3.0)
        c = get_counters()
        print(
            f"[MQTT Final Counters] Received: {c['received']} | Accepted: {c['accepted']} | "
            f"Rejected: {c['rejected']} | Written: {c['written']} | Write Failed: {c['write_failed']}"
        )


def main():
    service = MqttIngestService()

    def handle_signal(sig, frame):
        service.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    service.start()
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
