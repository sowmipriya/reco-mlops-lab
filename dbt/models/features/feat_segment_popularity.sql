-- Top products per customer segment: fallback for customers with little history.
with scored as (
    select
        c.segment,
        e.product_id,
        sum(e.weight) as score
    from {{ ref('fct_interactions') }} e
    join {{ ref('dim_customers') }} c using (customer_id)
    join {{ ref('dim_products') }} p using (product_id)
    where p.is_active
    group by 1, 2
)

select *
from (
    select
        segment,
        product_id,
        score,
        row_number() over (partition by segment order by score desc) as rank_in_segment
    from scored
)
where rank_in_segment <= 100
