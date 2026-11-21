with promos as (
    select
        product_id,
        count(*)            as n_promotions,
        max(discount_pct)   as max_discount_pct
    from {{ ref('stg_promotions') }}
    group by product_id
)

select
    p.product_id,
    p.product_name,
    p.category,
    p.subcategory,
    p.brand,
    p.price_eur,
    p.is_active,
    coalesce(pr.n_promotions, 0)  as n_promotions,
    pr.max_discount_pct
from {{ ref('stg_products') }} p
left join promos pr using (product_id)
