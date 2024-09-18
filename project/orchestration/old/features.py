from feast import Entity, Feature, FeatureView, ValueType
from feast.data_sources import FileSource
from datetime import timedelta

# Define la entidad
trip = Entity(name="trip_id", value_type=ValueType.STRING, description="Taxi trip ID")

# Define la fuente de datos
trip_source = FileSource(
    path="data/normalized_features.parquet",
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created",
)

# Define las características
trip_features = FeatureView(
    name="trip_features",
    entities=["trip_id"],
    ttl=timedelta(days=1),
    features=[
        Feature(name="PU_DO", dtype=ValueType.STRING),
        Feature(name="trip_distance", dtype=ValueType.FLOAT),
        Feature(name="duration", dtype=ValueType.FLOAT),
    ],
    batch_source=trip_source,
    online=True,
)
