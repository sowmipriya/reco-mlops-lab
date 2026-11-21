-- Pre-built customer features served by the online feature store.
-- "as of" the last event in the data (the synthetic timeline is frozen).
with events as (
    select * from {{ ref('fct_interactions') }}
),

as_of as (
    select max(event_ts) as as_of_ts from events
),

per_product as (
    select
        customer_id,
        product_id,
        sum(weight)    as weight,
        max(event_ts)  as last_ts
    from events
    group by customer_id, product_id
),

recent as (
    select
        customer_id,
        array_agg(product_id order by last_ts desc limit 20)               as recent_products,
        array_agg(struct(product_id, weight) order by weight desc limit 20) as top_products
    from per_product
    group by customer_id
),

category_affinity as (
    select
        e.customer_id,
        array_agg(p.category order by w desc limit 1)[offset(0)] as favorite_category
    from (
        select customer_id, product_id, sum(weight) as w
        from events group by 1, 2
    ) e
    join {{ ref('dim_products') }} p using (product_id)
    group by e.customer_id
),

activity as (
    select
        e.customer_id,
        countif(e.event_ts >= timestamp_sub(a.as_of_ts, interval 30 day))                              as n_events_30d,
        countif(e.event_type = 'purchase' and e.event_ts >= timestamp_sub(a.as_of_ts, interval 90 day)) as n_purchases_90d,
        max(e.event_ts)                                                                                 as last_event_ts
    from events e
    cross join as_of a
    group by e.customer_id
)

select
    c.customer_id,
    c.segment,
    c.region,
    c.loyalty_member,
    coalesce(act.n_events_30d, 0)     as n_events_30d,
    coalesce(act.n_purchases_90d, 0)  as n_purchases_90d,
    act.last_event_ts,
    ca.favorite_category,
    coalesce(r.recent_products, [])   as recent_products,
    r.top_products,
    (select as_of_ts from as_of)      as feature_ts
from {{ ref('dim_customers') }} c
left join activity act using (customer_id)
left join category_affinity ca using (customer_id)
left join recent r using (customer_id)
