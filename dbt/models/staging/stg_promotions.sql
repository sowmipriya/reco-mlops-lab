select
    promotion_id,
    product_id,
    cast(discount_pct as int64) as discount_pct,
    timestamp(start_ts)         as start_ts,
    timestamp(end_ts)           as end_ts
from {{ source('raw', 'promotions') }}
