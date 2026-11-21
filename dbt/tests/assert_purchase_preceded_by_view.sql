-- Data contract with the producing domain: a purchase must follow a view or add-to-cart
-- of the same product by the same customer, in this or an EARLIER session (customers come
-- back to buy). Real tracking loses a few events, so the contract tolerates up to 1%:
-- the test returns a row (fails) only when the violation rate is above that threshold.
with purchases as (
    select event_id, customer_id, product_id, event_ts
    from {{ ref('fct_interactions') }}
    where event_type = 'purchase'
),

seen_before as (
    select distinct p.event_id
    from purchases p
    join {{ ref('fct_interactions') }} e
      on e.customer_id = p.customer_id
     and e.product_id = p.product_id
     and e.event_type in ('view', 'add_to_cart')
     and e.event_ts <= p.event_ts
)

select
    count(*)                                              as n_purchases,
    countif(s.event_id is null)                           as n_violations,
    safe_divide(countif(s.event_id is null), count(*))    as violation_rate
from purchases p
left join seen_before s using (event_id)
having violation_rate > 0.01
