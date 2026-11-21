{{
  config(
    partition_by={"field": "event_ts", "data_type": "timestamp", "granularity": "day"},
    cluster_by=["customer_id", "product_id"]
  )
}}

{%- set w = var('event_weights') %}

-- One row per behavioural event, enriched with the implicit-feedback weight
-- and the train/test split used for offline evaluation.
with events as (
    select * from {{ ref('stg_events') }}
),

bounds as (
    select timestamp_sub(max(event_ts), interval {{ var('test_days') }} day) as split_ts
    from events
)

select
    e.event_id,
    e.customer_id,
    e.session_id,
    e.product_id,
    e.event_type,
    e.event_ts,
    e.quantity,
    e.channel,
    cast(case e.event_type
        when 'view' then {{ w['view'] }}
        when 'add_to_cart' then {{ w['add_to_cart'] }}
        when 'purchase' then {{ w['purchase'] }}
    end as float64)                                               as weight,
    if(e.event_ts < b.split_ts, 'train', 'test')                  as split
from events e
cross join bounds b
