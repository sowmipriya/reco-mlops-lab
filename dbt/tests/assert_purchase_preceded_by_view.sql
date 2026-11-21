-- Data contract with the producing domain: every purchase must follow a view of
-- the same product in the same session. Returns violating purchases.
select p.event_id
from {{ ref('fct_interactions') }} p
left join {{ ref('fct_interactions') }} v
  on v.session_id = p.session_id
 and v.product_id = p.product_id
 and v.event_type = 'view'
 and v.event_ts <= p.event_ts
where p.event_type = 'purchase'
  and v.event_id is null
