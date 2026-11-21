-- Customer dimension. The public data has no customer attributes, so the segment is
-- behavioural: the department the customer engaged with most during the TRAIN period only.
-- Using train events only keeps the offline evaluation point-in-time correct.
{%- set w = var('event_weights') %}

with train_events as (
    select
        e.customer_id,
        p.category,
        case e.event_type
            when 'view' then {{ w['view'] }}
            when 'add_to_cart' then {{ w['add_to_cart'] }}
            when 'purchase' then {{ w['purchase'] }}
        end as weight
    from {{ ref('stg_events') }} e
    join {{ ref('stg_products') }} p using (product_id)
    where e.event_ts < (
        select timestamp_sub(max(event_ts), interval {{ var('test_days') }} day) from {{ ref('stg_events') }}
    )
),

main_department as (
    select
        customer_id,
        array_agg(category order by w desc, category limit 1)[offset(0)] as segment
    from (select customer_id, category, sum(weight) as w from train_events group by 1, 2)
    group by customer_id
)

select
    c.customer_id,
    c.first_seen_ts,
    coalesce(m.segment, 'new_customer') as segment
from {{ ref('stg_customers') }} c
left join main_department m using (customer_id)
