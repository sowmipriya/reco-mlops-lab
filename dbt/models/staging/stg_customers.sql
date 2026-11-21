select
    customer_id,
    segment,
    region,
    loyalty_member,
    timestamp(signup_ts) as signup_ts
from {{ source('raw', 'customers') }}
