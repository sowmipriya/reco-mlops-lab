-- Time-decayed popularity (half-life ~10 days) used for cold start and
-- category-level recommendations.
with as_of as (
    select max(event_ts) as as_of_ts from {{ ref('fct_interactions') }}
),

scores as (
    select
        e.product_id,
        sum(e.weight * exp(-timestamp_diff(a.as_of_ts, e.event_ts, hour) / 24.0 / 14.0)) as popularity_score,
        countif(e.event_type = 'purchase')                                               as n_purchases
    from {{ ref('fct_interactions') }} e
    cross join as_of a
    group by e.product_id
)

select
    p.product_id,
    p.category,
    p.subcategory,
    coalesce(s.popularity_score, 0)   as popularity_score,
    coalesce(s.n_purchases, 0)        as n_purchases,
    row_number() over (partition by p.category order by coalesce(s.popularity_score, 0) desc) as rank_in_category
from {{ ref('dim_products') }} p
left join scores s using (product_id)
where p.is_active
