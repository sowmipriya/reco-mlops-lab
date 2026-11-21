-- Evaluation ground truth: products a customer added to cart or bought
-- during the held-out window.
select distinct
    customer_id,
    product_id
from {{ ref('fct_interactions') }}
where split = 'test'
  and event_type in ('add_to_cart', 'purchase')
