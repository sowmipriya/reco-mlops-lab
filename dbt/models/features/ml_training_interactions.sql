-- Training set: customer x product implicit-feedback matrix, train period only
-- (point-in-time correct: nothing from the evaluation window leaks in).
select
    customer_id,
    product_id,
    sum(weight)    as weight,
    count(*)       as n_events,
    max(event_ts)  as last_ts
from {{ ref('fct_interactions') }}
where split = 'train'
group by customer_id, product_id
