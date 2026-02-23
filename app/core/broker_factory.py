"""
Broker factory: returns Kafka or RabbitMQ broker based on MESSAGE_BROKER.
"""
from __future__ import annotations

from app.config import settings
from faststream.kafka.fastapi import KafkaRouter
from faststream.rabbit.fastapi import RabbitRouter

def _build_rabbitmq_url() -> str:
    if settings.RABBITMQ_URL:
        return settings.RABBITMQ_URL
    username = settings.RABBITMQ_USERNAME or "guest"
    password = settings.RABBITMQ_PASSWORD or "guest"
    host = settings.RABBITMQ_HOST or "localhost"
    port = settings.RABBITMQ_PORT or 5672
    vhost = (settings.RABBITMQ_VHOST or "/").lstrip("/")
    return f"amqp://{username}:{password}@{host}:{port}/{vhost}"


def get_broker():
    """Return Kafka or RabbitMQ broker based on MESSAGE_BROKER."""
    if settings.MESSAGE_BROKER == "kafka":
        brokers = [b.strip() for b in settings.KAFKA_BROKERS.split(",")]
        return KafkaRouter(
            bootstrap_servers=brokers,
            client_id=settings.KAFKA_CLIENT_ID,
        )

    return RabbitRouter(url=_build_rabbitmq_url())
